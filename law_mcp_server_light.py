#!/usr/bin/env python3
"""
A2AJ Law Database MCP Server - Lightweight Version
Uses httpx directly instead of supabase SDK to avoid C++ build dependencies.
"""

import os
import json
import logging
from typing import Any, Optional
from dataclasses import dataclass
from datetime import datetime

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    "https://fxnlejuzrdeywoinctat.supabase.co"
)

# Check both env var names for compatibility (render_blueprint uses SUPABASE_KEY)
# .strip() guards against trailing whitespace from shell env injection
SUPABASE_KEY = (
    os.environ.get("SUPABASE_KEY", "").strip() or
    os.environ.get("SUPABASE_ANON_KEY", "").strip() or
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ4bmxlanV6cmRleXdvaW5jdGF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjczMzc4MzQsImV4cCI6MjA4MjkxMzgzNH0.VsDYjaqy-6c_1r8csGJFueaTUYqxGpNfnIf0DhOQGE0"
)

# =============================================================================
# HTTP Client
# =============================================================================

class SupabaseClient:
    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
    
    def table(self, table_name: str):
        return TableQuery(self, table_name)

@dataclass
class QueryBuilder:
    client: SupabaseClient
    table: str
    select_cols: str = "*"
    filters: list = None
    order_col: str = None
    order_desc: bool = False
    limit_val: int = None
    offset_val: int = None
    
    def __post_init__(self):
        self.filters = self.filters or []
    
    def select(self, cols: str):
        self.select_cols = cols
        return self
    
    def eq(self, column: str, value: str):
        self.filters.append({"column": column, "operator": "eq", "value": value})
        return self
    
    def ilike(self, column: str, pattern: str):
        self.filters.append({"column": column, "operator": "ilike", "value": pattern})
        return self
    
    def order(self, column: str, desc: bool = False):
        self.order_col = column
        self.order_desc = desc
        return self
    
    def limit(self, n: int):
        self.limit_val = n
        return self
    
    def offset(self, n: int):
        self.offset_val = n
        return self
    
    def execute(self):
        # Build URL
        url = f"{self.client.url}/rest/v1/{self.table}"
        params = {}
        
        # Select
        params["select"] = self.select_cols
        
        # Filters - use PostgREST filter syntax as separate params
        for f in self.filters:
            col = f["column"]
            op = f["operator"]
            val = f["value"]
            if op == "eq":
                # Use direct query param: column=eq.value
                params[col] = f"eq.{val}"
            elif op == "ilike":
                # PostgREST uses * as wildcard (not SQL %).
                # Convert any SQL % wildcards from callers to PostgREST *
                postgrest_val = val.replace("%", "*")
                params[col] = f"ilike.{postgrest_val}"
        
        # Order
        if self.order_col:
            asc_desc = "desc" if self.order_desc else "asc"
            params["order"] = f"{self.order_col}.{asc_desc}"
        
        # Limit/Offset
        if self.limit_val:
            params["limit"] = self.limit_val
        if self.offset_val:
            params["offset"] = self.offset_val
        
        response = httpx.get(url, headers=self.client.headers, params=params)
        response.raise_for_status()
        return type('Response', (), {'data': response.json()})()

class TableQuery:
    def __init__(self, client: SupabaseClient, table: str):
        self.client = client
        self.table = table
    
    def select(self, cols: str = "*"):
        return QueryBuilder(self.client, self.table).select(cols)

# =============================================================================
# MCP Tool Classes
# =============================================================================

@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    input_schema: dict
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }

@dataclass  
class MCPToolResult:
    """Result from an MCP tool."""
    content: list
    is_error: bool = False
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "isError": self.is_error
        }

# =============================================================================
# Tool Implementations
# =============================================================================

def get_supabase() -> SupabaseClient:
    return SupabaseClient(SUPABASE_URL, SUPABASE_KEY)

async def search_cases(
    query: str,
    jurisdiction: Optional[str] = None,
    dataset: Optional[str] = None,
    limit: int = 10
) -> MCPToolResult:
    """Search case law using keyword search."""
    supabase = get_supabase()
    
    try:
        db_query = supabase.table("case_law").select(
            "id, citation_en, name_en, dataset, document_date_en, unofficial_text_en"
        )
        
        if jurisdiction:
            db_query = db_query.eq("dataset", jurisdiction)
        if dataset:
            db_query = db_query.eq("dataset", dataset)
        
        if query:
            db_query = db_query.ilike("name_en", f"%{query}%")
        
        response = db_query.limit(limit).execute()
        
        results = []
        for case in response.data:
            text = case.get("unofficial_text_en", "") or ""
            excerpt = text[:500] + "..." if len(text) > 500 else text
            
            results.append({
                "citation": case.get("citation_en", "N/A"),
                "name": case.get("name_en", "N/A"),
                "court": case.get("dataset", "N/A"),
                "date": str(case.get("document_date_en", ""))[:10] if case.get("document_date_en") else "N/A",
                "excerpt": excerpt
            })
        
        if not results:
            return MCPToolResult([
                {"type": "text", "text": f"No cases found for query: {query}"}
            ])
        
        output = f"Found {len(results)} cases:\n\n"
        for i, case in enumerate(results, 1):
            output += f"{i}. **{case['citation']}** - {case['name']}\n"
            output += f"   Court: {case['court']} | Date: {case['date']}\n"
            output += f"   {case['excerpt'][:200]}...\n\n"
        
        return MCPToolResult([{"type": "text", "text": output}])
        
    except Exception as e:
        return MCPToolResult([
            {"type": "text", "text": f"Error searching cases: {str(e)}"}
        ], is_error=True)


async def search_legislation(
    query: str,
    jurisdiction: Optional[str] = None,
    limit: int = 10
) -> MCPToolResult:
    """Search legislation by keyword."""
    supabase = get_supabase()
    
    try:
        db_query = supabase.table("a2aj_legislation_sections").select(
            "id, act_citation, act_title_en, jurisdiction, section_number, section_text_en"
        )
        
        if jurisdiction:
            db_query = db_query.eq("jurisdiction", jurisdiction)
        
        if query:
            db_query = db_query.ilike("act_title_en", f"%{query}%")
        
        response = db_query.limit(limit).execute()
        
        results = []
        for item in response.data:
            text = item.get("section_text_en", "") or ""
            excerpt = text[:300] + "..." if len(text) > 300 else text
            
            results.append({
                "citation": item.get("act_citation", "N/A"),
                "title": item.get("act_title_en", "N/A"),
                "jurisdiction": item.get("jurisdiction", "N/A"),
                "section": item.get("section_number", "N/A"),
                "excerpt": excerpt
            })
        
        if not results:
            return MCPToolResult([
                {"type": "text", "text": f"No legislation found for query: {query}"}
            ])
        
        output = f"Found {len(results)} legislation results:\n\n"
        for i, item in enumerate(results, 1):
            output += f"{i}. **{item['title']}** ({item['jurisdiction']})\n"
            output += f"   Citation: {item['citation']} §{item['section']}\n"
            output += f"   {item['excerpt'][:150]}...\n\n"
        
        return MCPToolResult([{"type": "text", "text": output}])
        
    except Exception as e:
        return MCPToolResult([
            {"type": "text", "text": f"Error searching legislation: {str(e)}"}
        ], is_error=True)


async def get_case_by_citation(citation: str) -> MCPToolResult:
    """Get a specific case by its citation."""
    supabase = get_supabase()
    
    try:
        response = supabase.table("case_law").select(
            "id, citation_en, name_en, dataset, document_date_en, url_en, unofficial_text_en"
        ).ilike("citation_en", f"%{citation}%").limit(1).execute()
        
        if not response.data:
            return MCPToolResult([
                {"type": "text", "text": f"Case not found: {citation}"}
            ], is_error=True)
        
        case = response.data[0]
        text = case.get("unofficial_text_en", "") or "No text available"
        
        output = f"## {case['citation_en']}\n"
        output += f"**{case['name_en']}**\n\n"
        output += f"**Court:** {case['dataset']}\n"
        output += f"**Date:** {str(case.get('document_date_en', ''))[:10]}\n"
        output += f"**URL:** {case.get('url_en', 'N/A')}\n\n"
        output += f"### Full Text:\n{text[:5000]}"
        
        return MCPToolResult([{"type": "text", "text": output}])
        
    except Exception as e:
        return MCPToolResult([
            {"type": "text", "text": f"Error getting case: {str(e)}"}
        ], is_error=True)


async def get_legislation_by_citation(
    citation: str,
    jurisdiction: Optional[str] = None
) -> MCPToolResult:
    """Get legislation by citation."""
    supabase = get_supabase()
    
    try:
        db_query = supabase.table("a2aj_legislation_sections").select(
            "act_citation, act_title_en, jurisdiction, section_number, section_text_en, source_url"
        ).ilike("act_citation", f"%{citation}%")
        
        if jurisdiction:
            db_query = db_query.eq("jurisdiction", jurisdiction)
        
        response = db_query.limit(50).execute()
        
        if not response.data:
            return MCPToolResult([
                {"type": "text", "text": f"Legislation not found: {citation}"}
            ], is_error=True)
        
        acts = {}
        for item in response.data:
            act_cit = item['act_citation']
            if act_cit not in acts:
                acts[act_cit] = {
                    "title": item['act_title_en'],
                    "jurisdiction": item['jurisdiction'],
                    "url": item.get('source_url', ''),
                    "sections": []
                }
            acts[act_cit]["sections"].append({
                "number": item['section_number'],
                "text": item.get('section_text_en', '')[:1000]
            })
        
        output = f"## {citation}\n\n"
        for act_cit, act_data in acts.items():
            output += f"### {act_data['title']} ({act_data['jurisdiction']})\n"
            output += f"URL: {act_data['url']}\n\n"
            for sec in act_data['sections'][:10]:
                output += f"**§ {sec['number']}**\n{sec['text']}\n\n"
        
        return MCPToolResult([{"type": "text", "text": output}])
        
    except Exception as e:
        return MCPToolResult([
            {"type": "text", "text": f"Error getting legislation: {str(e)}"}
        ], is_error=True)


async def list_jurisdictions() -> MCPToolResult:
    """List all available jurisdictions."""
    supabase = get_supabase()
    
    try:
        # Get unique jurisdictions from legislation
        leg_response = supabase.table("a2aj_legislation_sections").select(
            "jurisdiction"
        ).limit(1000).execute()
        
        # Get unique jurisdictions from cases  
        case_response = supabase.table("case_law").select(
            "dataset"
        ).limit(1000).execute()
        
        jurisdictions = set()
        for item in leg_response.data:
            j = item.get('jurisdiction')
            if j:
                jurisdictions.add(j)
        
        for item in case_response.data:
            j = item.get('dataset')
            if j:
                jurisdictions.add(j)
        
        output = "## Available Jurisdictions\n\n"
        output += "| Jurisdiction |\n"
        output += "|-------------|\n"
        
        for j in sorted(jurisdictions):
            output += f"| {j} |\n"
        
        output += f"\n**Total: {len(jurisdictions)} jurisdictions**"
        
        return MCPToolResult([{"type": "text", "text": output}])
        
    except Exception as e:
        return MCPToolResult([
            {"type": "text", "text": f"Error listing jurisdictions: {str(e)}"}
        ], is_error=True)


async def get_cases_by_jurisdiction(
    jurisdiction: str,
    limit: int = 20
) -> MCPToolResult:
    """Get recent cases from a jurisdiction."""
    supabase = get_supabase()
    
    try:
        response = supabase.table("case_law").select(
            "citation_en, name_en, dataset, document_date_en"
        ).eq("dataset", jurisdiction).order(
            "document_date_en", desc=True
        ).limit(limit).execute()
        
        if not response.data:
            return MCPToolResult([
                {"type": "text", "text": f"No cases found for jurisdiction: {jurisdiction}"}
            ])
        
        output = f"## Recent Cases - {jurisdiction}\n\n"
        for i, case in enumerate(response.data, 1):
            output += f"{i}. **{case['citation_en']}** - {case['name_en']}\n"
            output += f"   Date: {str(case.get('document_date_en', ''))[:10]}\n\n"
        
        return MCPToolResult([{"type": "text", "text": output}])
        
    except Exception as e:
        return MCPToolResult([
            {"type": "text", "text": f"Error getting cases: {str(e)}"}
        ], is_error=True)


# =============================================================================
# MCP Server Setup
# =============================================================================

TOOLS = [
    MCPTool(
        name="search_cases",
        description="Search Canadian case law by keyword.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "jurisdiction": {"type": "string", "description": "Filter by jurisdiction (optional)"},
                "dataset": {"type": "string", "description": "Filter by court (optional)"},
                "limit": {"type": "integer", "description": "Number of results (default 10)"}
            },
            "required": ["query"]
        }
    ),
    MCPTool(
        name="search_legislation",
        description="Search Canadian legislation by keyword.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "jurisdiction": {"type": "string", "description": "Filter by jurisdiction (optional)"},
                "limit": {"type": "integer", "description": "Number of results (default 10)"}
            },
            "required": ["query"]
        }
    ),
    MCPTool(
        name="get_case_by_citation",
        description="Get full details of a case by citation.",
        input_schema={
            "type": "object",
            "properties": {
                "citation": {"type": "string", "description": "Case citation"}
            },
            "required": ["citation"]
        }
    ),
    MCPTool(
        name="get_legislation_by_citation",
        description="Get full text of legislation by citation.",
        input_schema={
            "type": "object",
            "properties": {
                "citation": {"type": "string", "description": "Legislation citation"},
                "jurisdiction": {"type": "string", "description": "Jurisdiction (optional)"}
            },
            "required": ["citation"]
        }
    ),
    MCPTool(
        name="list_jurisdictions",
        description="List all available Canadian jurisdictions.",
        input_schema={"type": "object", "properties": {}}
    ),
    MCPTool(
        name="get_cases_by_jurisdiction",
        description="Get recent cases from a jurisdiction.",
        input_schema={
            "type": "object",
            "properties": {
                "jurisdiction": {"type": "string", "description": "Jurisdiction name"},
                "limit": {"type": "integer", "description": "Number of results (default 20)"}
            },
            "required": ["jurisdiction"]
        }
    )
]

TOOL_HANDLERS = {
    "search_cases": search_cases,
    "search_legislation": search_legislation,
    "get_case_by_citation": get_case_by_citation,
    "get_legislation_by_citation": get_legislation_by_citation,
    "list_jurisdictions": list_jurisdictions,
    "get_cases_by_jurisdiction": get_cases_by_jurisdiction
}

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="A2AJ Law MCP Server")

# CORS middleware - allow all origins for MCP bridge access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key auth is optional: only enforced if MCP_API_KEY is explicitly set
_raw_api_key = os.environ.get("MCP_API_KEY", "")
MCP_API_KEY = _raw_api_key if _raw_api_key and _raw_api_key != "your-secret-key-change-this" else None

@app.get("/")
async def root():
    return {"name": "A2AJ Law MCP Server", "version": "1.0.0", "status": "running"}

@app.get("/tools")
async def list_tools():
    return {"tools": [t.to_dict() for t in TOOLS]}

@app.post("/tools/call")
async def call_tool(request: Request):
    # Optional API key check — only enforced when MCP_API_KEY env var is set
    if MCP_API_KEY:
        api_key = request.headers.get("X-API-Key")
        if api_key != MCP_API_KEY:
            logger.warning("Rejected request: invalid API key")
            return JSONResponse(
                status_code=401,
                content=MCPToolResult(
                    [{"type": "text", "text": "Invalid or missing API key"}], is_error=True
                ).to_dict()
            )

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        logger.error("Failed to parse request JSON")
        return JSONResponse(
            status_code=400,
            content=MCPToolResult(
                [{"type": "text", "text": "Invalid JSON in request body"}], is_error=True
            ).to_dict()
        )

    tool_name = body.get("name")
    arguments = body.get("arguments", {})

    if tool_name not in TOOL_HANDLERS:
        logger.warning(f"Tool not found: {tool_name}")
        return JSONResponse(
            status_code=404,
            content={"detail": f"Tool not found: {tool_name}"}
        )

    # Execute the tool handler
    try:
        logger.info(f"Calling tool: {tool_name} with args: {arguments}")
        handler = TOOL_HANDLERS[tool_name]
        result = await handler(**arguments)
        logger.info(f"Tool {tool_name} completed successfully")
        return JSONResponse(content=result.to_dict())
    except Exception as e:
        logger.exception(f"Error executing tool {tool_name}: {e}")
        return JSONResponse(
            status_code=500,
            content=MCPToolResult(
                [{"type": "text", "text": f"Server error: {str(e)}"}], is_error=True
            ).to_dict()
        )

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("=" * 60)
    print("A2AJ Law Database MCP Server")
    print("Lightweight Version (httpx)")
    print("=" * 60)
    print(f"Supabase: {SUPABASE_URL}")
    print(f"API Key auth: {'ENABLED' if MCP_API_KEY else 'DISABLED (open access)'}")
    print("\nTools available:")
    for tool in TOOLS:
        print(f"  - {tool.name}")
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=port)
