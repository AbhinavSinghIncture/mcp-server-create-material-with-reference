# =============================================================================
# IMPORTS - Standard Library
# =============================================================================
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Annotated, List, Dict, Any, Optional, Tuple

# =============================================================================
# IMPORTS - Third Party Libraries
# =============================================================================
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import FastMCP
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowState(TypedDict):
    #Authorization for API calls
    authorization:str

    # Core communication
    messages: Annotated[list, add_messages]
    
    # Search parameters from user
    material_type: Optional[str]
    material_number: Optional[str] 
    plant: Optional[str]
    requestId: Optional[str]
    
    # Search results
    search_results: List[Dict[str, Any]]  # Raw DB results
    formatted_results: List[Dict[str, str]]  # Formatted for user display
    selected_item_index: Optional[int]
    
    # Status tracking
    search_status: str  # "searching", "results_found", "no_results", "selected", "completed"
    needs_refinement: bool
    
    # Final output for integration with your main workflow
    reference_data: Optional[Dict[str, Any]]  # Complete material data
    fieldNames: List[Dict[str, str]]  # For your WorkflowState integration

    #For Maintaining configurable thread id value for user isolation
    user_email:Optional[str]

# =============================================================================
# SECTION 3: HELPER FUNCTIONS
# =============================================================================
def get_authorization_from_env() -> str:
    """
    Helper method to get authorization token from environment variable
    
    Returns:
        str: Authorization token from AUTHROISATION_TOKEN env variable
        
    Raises:
        ValueError: If token not found in environment
    """
    token = os.getenv("AUTHORISATION_TOKEN", "")
    
    if not token:
        raise ValueError("Authorization token not found in environment variable 'AUTHROISATION_TOKEN'")
    
    return token

def add_authorization_to_state(state: WorkflowState) -> WorkflowState:
    """
    Add authorization token to existing state
    """
    auth_token = get_authorization_from_env()
    if(auth_token!=""):
        print("AUTH TOKEN RECEIVED FROM ENV")
    return {
        **state,
        "authorization": auth_token
    }
def initialize_gemini_llm():
    """Simple method to initialize Gemini LLM using LangChain."""
    global llm  # ✅ Add this line to make it global
    
    load_dotenv() 
    # Get API key from env
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Stable and fast
        api_key=api_key,  # Use the variable, not os.getenv again
        temperature=0.1
    )
    
    logger.info("✅ Global LLM initialized successfully")
    return llm

# ✅ Initialize the LLM globally when module loads
llm = None  # Declare global variable first
initialize_gemini_llm()  # ✅ ADD THIS LINE - Actually call the function!

import requests
import json
from typing import Dict, Any, Optional

def call_api(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Generic API call template
    """
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=payload,
            params=params,
            timeout=timeout
        )
        response.raise_for_status()
        return {
            "success": True,
            "data": response.json(),
            "status_code": response.status_code
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None)
        }


from langchain_core.messages import HumanMessage, SystemMessage
import json
import re
from typing import List, Dict


def extract_material_search_fields(user_message: str, llm) -> List[Dict[str, str]]:
    """
    Use LLM to extract material reference search fields from user message
    
    Args:
        user_message: User's query text
        llm: Initialized LLM instance
        
    Returns:
        List of field-value dictionaries for material search
    """
    try:
        context = """
        You are an expert at extracting material search parameters from user queries for SAP Material Reference Search.
        
        TASK: Extract ONLY these 4 field types and their values from the user's message:
        
        1. REQUEST ID FIELDS (standardize as "request_id"):
           - request_id, requestId, reqId, req_id, request, req
           - "request REQ-001", "reqId: REQ-123", "use request REQ-456"
        
        2. MATERIAL TYPE FIELDS (standardize as "material_type"):  
           - material_type, materialType, matltype, matl_type, type, category
           - "FERT", "ROH", "HALB", "DIEN", "NLAG"
           - "FERT materials", "type FERT", "material type ROH"
        
        3. MATERIAL NUMBER FIELDS (standardize as "material_number"):
           - material_number, materialNumber, material, matl, mat_no, material_no
           - "FERT1001", "material FERT1250", "matl: ABC123"
        
        4. PLANT FIELDS (standardize as "plant"):
           - plant, plants, plant_code, plantCode, facility, location, site
           - "1001", "plant 1001", "facility 2001", "site 3001"
        
        INSTRUCTIONS:
        1. Find ONLY these 4 field types in the user's message
        2. Return ONLY a JSON array with "fieldName" and "value" keys
        3. Use EXACT standardized names: "request_id", "material_type", "material_number", "plant"
        4. If no relevant fields found, return empty array: []
        5. DO NOT wrap response in markdown code blocks
        6. Return ONLY the raw JSON array
        
        EXAMPLES:
        Input: "Search for FERT materials in plant 1001"
        Output: [{"fieldName": "material_type", "value": "FERT"}, {"fieldName": "plant", "value": "1001"}]
        
        Input: "Find material FERT1250 from facility 2001"  
        Output: [{"fieldName": "material_number", "value": "FERT1250"}, {"fieldName": "plant", "value": "2001"}]
        
        Input: "Use request REQ-001 for reference"
        Output: [{"fieldName": "request_id", "value": "REQ-001"}]
        
        INPUT: "Show FERT type materials"
        Output: [{"fieldName": "material_type", "value": "FERT"}]
        
        USER MESSAGE TO ANALYZE:
        """
        
        system_msg = SystemMessage(content=context)
        human_msg = HumanMessage(content=user_message)
        
        response = llm.invoke([system_msg, human_msg])
        llm_response = response.content.strip()
        
        json_str = strip_markdown_json(llm_response)
        
        try:
            extracted_fields = json.loads(json_str)
            if isinstance(extracted_fields, list):
                valid_fields = []
                allowed_fields = ["request_id", "material_type", "material_number", "plant"]
                
                for field in extracted_fields:
                    if isinstance(field, dict) and "fieldName" in field and "value" in field:
                        field_name = str(field["fieldName"]).lower()
                        field_value = str(field["value"]).strip()
                        
                        if field_name in allowed_fields and field_value:
                            valid_fields.append({
                                "fieldName": field_name,
                                "value": field_value
                            })
                
                print(f"Response: {valid_fields}")
                return valid_fields
            else:
                print("Response: []")
                return []
                
        except json.JSONDecodeError:
            print("Response: []")
            return []
            
    except Exception:
        print("Response: []")
        return []


def strip_markdown_json(text: str) -> str:
    """
    Strip markdown formatting from JSON response
    
    Args:
        text: Raw LLM response that may contain markdown
        
    Returns:
        Clean JSON string
    """
    pattern = r'``````'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return text.strip()

def build_material_search_payload(state: WorkflowState) -> Dict[str, Any]:
    """
    Build API payload from WorkflowState variables
    
    Args:
        state: WorkflowState containing search parameters
        
    Returns:
        Payload dict with 4 keys mapped from state
    """
    payload = {}
    
    # Map WorkflowState variables to API payload keys
    if state.get("material_number"):
        payload["material"] = state["material_number"]
    else :
        payload["material"]=""
    
    if state.get("material_type"):
        payload["materialType"] = state["material_type"]
    else :
        payload["materialType"]=""
    if state.get("plant"):
        payload["plant"] = state["plant"]
    else :
        payload["plant"]=""
    if state.get("requestId"):
        payload["requestId"] = state["requestId"]
    else :
        payload["requestId"]=""
    
    print(f"🔧 Built API payload: {payload}")
    return payload

def search_materials_api_call(state: WorkflowState, api_url: str = None) -> Dict[str, Any]:  # ✅ Changed return type
    """
    Make API call to search materials endpoint using generic call_api method
    
    Args:
        state: WorkflowState containing search parameters
        api_url: API endpoint URL (to be defined later)
        
    Returns:
        Full API response dict with statusCode, message, body, count
    """
    try:
        # Use placeholder URL if not provided (to be defined later)
        if not api_url:
            api_url = "https://incture-cherrywork-dev-cw-mdg-dev-cw-mdg-materialmanagement.cfapps.eu10-004.hana.ondemand.com/data/fetchCreateWithReferenceMaterialWithAiSearchResults"
        
        # Build payload from WorkflowState
        payload = build_material_search_payload(state)
        
        # Validate payload has at least one search parameter
        if not payload:
            print("⚠️ No search parameters provided in payload")
            return {}  # ✅ Return empty dict instead of empty list
        
        # Use authorization token from state
        headers = {
            "Content-Type": "application/json",
            "Authorization": state.get("authorization", "")
        }
        
        print(f"🌐 Making API call to: {api_url}")
        print(f"📦 Request payload: {json.dumps(payload, indent=2)}")
        
        # Make API call using generic call_api method
        api_response = call_api(
            url=api_url,
            method="POST",
            headers=headers,
            payload=payload,
            timeout=30
        )
        
        # Print API response
        print(f"📡 API Response Status: {api_response.get('success')}")
        print(f"📡 Status Code: {api_response.get('status_code')}")
        print(f"📡 Body : {api_response.get('data')}")

        if api_response.get("success"):
            response_data = api_response.get("data", {})
            print(f"📊 API Response Data: {json.dumps(response_data, indent=2)}")
            
            # ✅ Return the FULL response data instead of just the body
            return response_data  # This contains statusCode, message, body, count
        else:
            error_msg = api_response.get("error", "Unknown API error")
            print(f"❌ API call failed: {error_msg}")
            return {}  # ✅ Return empty dict instead of empty list
            
    except Exception as e:
        print(f"❌ Unexpected error in search_materials_api_call: {e}")
        return {}  # ✅ Return empty dict instead of empty list


# =============================================================================
# SECTION 4: WORKFLOW NODE FUNCTIONS
# =============================================================================

def material_search_extract_node(state: WorkflowState) -> WorkflowState:
    """Extract material reference search parameters from user message"""
    try:
        # Get the latest user message
        messages = state.get("messages", [])
        if not messages:
            return {
                **state,
                "search_status": "searching",
                "needs_refinement": True,
                "requestId":None,
                "material_type": None,
                "material_number": None,
                "plant": None
            }
        
        latest_message = messages[-1]
        user_text = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
        
        print(f"🔍 Analyzing message: {user_text}")
        
        # Use LLM to extract material search fields
        extracted_fields = extract_material_search_fields(user_text, llm)
        
        # Extract search parameters from fields
        material_type = None
        material_number = None
        plant = None
        request_id = None
        
        for field in extracted_fields:
            field_name = field.get("fieldName")
            field_value = field.get("value")
            
            if field_name == "material_type":
                material_type = field_value
            elif field_name == "material_number":
                material_number = field_value
            elif field_name == "plant":
                plant = field_value
            elif field_name == "request_id":
                request_id = field_value
        
        # Determine if we have enough parameters to search
        has_search_params = any([material_type, material_number, plant, request_id])
        needs_refinement = not has_search_params
        
        # Set search status based on extracted parameters
        if request_id:
            search_status = "searching"  # Direct request ID search
        elif has_search_params:
            search_status = "searching"  # Field-based search
        else:
            search_status = "searching"  # Need more parameters
        
        print(f"🔍 Extracted parameters:")
        print(f"   Material Type: {material_type}")
        print(f"   Material Number: {material_number}")
        print(f"   Plant: {plant}")
        print(f"   requestId: {request_id}")
        print(f"   Status: {search_status}")
        print(f"   Needs Refinement: {needs_refinement}")
        
        return {
            **state,
            "material_type": material_type,
            "material_number": material_number,
            "plant": plant,
            "search_status": search_status,
            "needs_refinement": needs_refinement,
            "requestId":request_id
        }
        
    except Exception as e:
        print(f"❌ Error in material_search_extract_node: {e}")
        return {
            **state,
            "search_status": "searching",
            "needs_refinement": True,
            "material_type": None,
            "material_number": None,
            "plant": None
        }

def material_search_node(state: WorkflowState, api_url: str = None) -> WorkflowState:
    """
    Check for requestId and make API call to search materials using generic call_api method
    """
    try:
        request_id = state.get("requestId")
        material_type = state.get("material_type")
        material_number = state.get("material_number")
        plant = state.get("plant")
        user_email = state.get("user_email")
        
        print(f"🔍 Starting material search for user: {user_email}")
        print(f"   Request ID: {request_id}")
        print(f"   Material Type: {material_type}")
        print(f"   Material Number: {material_number}")
        print(f"   Plant: {plant}")
        
        # Check if we have any search parameters
        if not any([request_id, material_type, material_number, plant]):
            print("⚠️ No search parameters provided")
            return {
                **state,
                "search_status": "no_results",
                "needs_refinement": True,
                "search_results": [],
                "formatted_results": []
            }
        
        # Make API call using the updated method
        api_response = search_materials_api_call(state, api_url)
        
        # Handle the specific API response format
        if not api_response:
            print("📭 API returned empty response")
            return {
                **state,
                "search_status": "no_results",
                "needs_refinement": True, 
                "search_results": [],
                "formatted_results": []
            }
        
        # Check if response has the expected structure
        if not isinstance(api_response, dict) or "body" not in api_response:
            print("⚠️ Invalid API response structure")
            return {
                **state,
                "search_status": "no_results",
                "needs_refinement": True,
                "search_results": [],
                "formatted_results": []
            }
        
        # Extract the actual results from the "body" field
        raw_results = api_response.get("body", [])
        status_code = api_response.get("statusCode", 0)
        message = api_response.get("message", "")
        count = api_response.get("count")
        
        print(f"📡 API Response - Status Code: {status_code}")
        print(f"📡 API Response - Message: {message}")
        print(f"📡 API Response - Count: {count}")
        print(f"📡 API Response - body: {raw_results}")

        
        if not raw_results or status_code != 200:
            print("📭 No materials found or API error")
            return {
                **state,
                "search_status": "no_results",
                "needs_refinement": True, 
                "search_results": [],
                "formatted_results": []
            }
        
        # Format results for display with correct field mappings
        formatted_results = []
        for i, result in enumerate(raw_results[:10], 1):  # Limit to 10 results
            formatted_result = {
                "index": str(i),
                "request_id": request_id or "",  # Maintain from state since not in response
                "material_number": result.get("material", ""),  # API field: "material"
                "material_type": result.get("materialType", ""),  # API field: "materialType"
                "plant": result.get("plant", ""),
                "description": f"{result.get('materialType', '')} Material",  # Generate description since not in API
                "status": "Active",  # Default since not in API response
                
                # Additional fields from API response
                "sales_org": result.get("salesOrg", ""),
                "distribution_channel": result.get("distributionChannel", ""),
                "storage_location": result.get("storageLocation", ""),
                "warehouse": result.get("warehouse", "")
            }
            formatted_results.append(formatted_result)
        
        print(f"✅ Found {len(raw_results)} materials")
        print("📋 Query Results:")
        for result in formatted_results:
            print(f"   {result['index']}. {result['material_number']} ({result['material_type']})")
            print(f"      Request: {result['request_id']}, Plant: {result['plant']}")
            print(f"      Sales Org: {result['sales_org']}, Storage: {result['storage_location']}")
        
        return {
            **state,
            "search_status": "results_found",
            "needs_refinement": False,
            "search_results": raw_results,  # Store original API response body
            "formatted_results": formatted_results
        }
        
    except Exception as e:
        print(f"❌ Error in material_search_node: {e}")
        return {
            **state,
            "search_status": "no_results",
            "needs_refinement": True,
            "search_results": [],
            "formatted_results": []
        }

# =============================================================================
# SECTION 5: STATEGRAPH CONSTRUCTION & COMPILATION
# =============================================================================

# Create the state graph
def create_material_reference_graph():
    """Create and compile the material reference search graph"""
    
    # Create graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("extract_parameters", material_search_extract_node)
    workflow.add_node("search_materials", material_search_node)
    workflow.add_node("complete", lambda state: {**state, "search_status": "completed"})
    # Add edges
    workflow.add_edge(START, "extract_parameters")
    
    # Conditional edge after parameter extraction
    def route_after_extraction(state):
        needs_refinement = state.get("needs_refinement", True)
        if needs_refinement:
            return "complete"  # End if no valid parameters
        else:
            return "search_materials"  # Continue to search
    
    workflow.add_conditional_edges(
        "extract_parameters",
        route_after_extraction,
        {
            "search_materials": "search_materials",
            "complete": "complete"
        }
    )
    
    # Conditional edge after search
    def route_after_search(state):
        search_status = state.get("search_status", "no_results")
        if search_status == "results_found":
            return "complete"  # End with results
        else:
            return "complete"  # End without results
    
    workflow.add_conditional_edges(
        "search_materials",
        route_after_search,
        {
            "complete": "complete"
        }
    )
    
    workflow.add_edge("complete", END)
    
    # Compile with memory
    app = workflow.compile()
    
    return app

def execute_material_search_tool(user_query: str):
    """Execute material search workflow and return results for MCP Tool 1"""
    
    # Create the workflow app
    material_reference_app = create_material_reference_graph()
    
    # Build initial state with the actual user query
    initial_state = {
        "authorization": get_authorization_from_env(),  # Add auth token
        "messages": [
            {"role": "user", "content": user_query}  # Use the passed user_query
        ],
        "material_type": None,
        "material_number": None,
        "plant": None,
        "requestId": None,
        "api_status_code": None,
        "api_message": None,
        "api_count": None,
        "search_results": [],
        "formatted_results": [],
        "selected_item_index": None,
        "search_status": "searching",
        "needs_refinement": True,
        "reference_data": None,
        "fieldNames": [],
        "user_email": None
    }
    
    try:
        # ✅ Simple invoke - no config needed
        final_state = material_reference_app.invoke(initial_state)
        
        # Prepare response based on workflow outcome
        search_status = final_state.get('search_status', 'unknown')
        formatted_results = final_state.get('formatted_results', [])
        
        if search_status == "completed" and formatted_results:
            # Success - return results for user selection
            return {
                "status": "results_found",
                "total_count": len(formatted_results),
                "results": formatted_results,
                "search_params": {
                    "material_type": final_state.get('material_type'),
                    "material_number": final_state.get('material_number'),
                    "plant": final_state.get('plant'),
                    "request_id": final_state.get('requestId')
                },
                "llm_instruction": f"Found {len(formatted_results)} material(s). " + 
                                 ("Ask user to select one by saying 'I want item X' where X is the number." 
                                  if len(formatted_results) > 1 
                                  else "Ask user to confirm: 'Do you want to proceed with this material?'")
            }
        
        elif search_status == "completed" and not formatted_results:
            # No results found
            return {
                "status": "no_results",
                "message": "No materials found with the provided search criteria",
                "search_params": {
                    "material_type": final_state.get('material_type'),
                    "material_number": final_state.get('material_number'), 
                    "plant": final_state.get('plant'),
                    "request_id": final_state.get('requestId')
                },
                "llm_instruction": "Tell user no materials were found. Suggest they refine their search criteria."
            }
        
        else:
            # Parameter extraction failed or other error
            return {
                "status": "needs_refinement",
                "message": "Could not extract valid search parameters from the query",
                "llm_instruction": "Ask user to provide clearer search criteria (material type, material number, plant, or request ID)."
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Workflow execution failed: {str(e)}",
            "llm_instruction": "Tell user there was a technical error. Ask them to try again."
        }

# Correct function call with proper query


# Tool 2 Implementation - Format for UI
def format_material_for_ui(selected_material: dict) -> str:
    """
    Format selected/confirmed material item into UI-parseable string format
    
    Args:
        selected_material: Single complete material object from search results
        
    Returns:
        Formatted string that UI can parse and use to trigger APIs
    """
    
    # Extract values from selected material
    request_id = selected_material.get('request_id', '')
    material_number = selected_material.get('material_number', '')
    material_type = selected_material.get('material_type', '')
    plant = selected_material.get('plant', '')
    sales_org = selected_material.get('sales_org', '')
    distribution_channel = selected_material.get('distribution_channel', '')
    storage_location = selected_material.get('storage_location', '')
    warehouse = selected_material.get('warehouse', '')
    
    # Create UI-parseable format - Starting with identifier text
    formatted_response = f"""MATERIAL_REFERENCE_READY_FOR_CREATION

    Selected Material Details for API Integration:
    ==================================================
    REQUEST_ID: {request_id}
    MATERIAL_NUMBER: {material_number}
    MATERIAL_TYPE: {material_type}
    PLANT: {plant}
    SALES_ORG: {sales_org}
    DISTRIBUTION_CHANNEL: {distribution_channel}
    STORAGE_LOCATION: {storage_location}
    WAREHOUSE: {warehouse}
    ==================================================

    UI_TRIGGER_DATA: {{"action": "create_with_reference", "material_number": "{material_number}", "material_type": "{material_type}", "plant": "{plant}", "request_id": "{request_id}", "sales_org": "{sales_org}", "distribution_channel": "{distribution_channel}", "storage_location": "{storage_location}", "warehouse": "{warehouse}"}}

    INSTRUCTIONS_FOR_UI: Parse the UI_TRIGGER_DATA JSON and trigger the material creation API with the provided reference values. The user has confirmed they want to create a new material using {material_number} as reference.
    """
    
    return formatted_response


# =============================================================================
# FASTMCP SERVER IMPLEMENTATION
# =============================================================================

from fastmcp import FastMCP
import json
from typing import Dict, Any

# Create FastMCP server instance
mcp = FastMCP("Material Search Server")

# =============================================================================
# RAW TOOL FUNCTIONS (for testing and decoration)
# =============================================================================

def _search_materials_for_reference(user_query: str) -> Dict[str, Any]:
    """
    Search for materials based on user query and return selectable results
    
    Args:
        user_query: User's search request
        
    Returns:
        Dict with search results and instructions for LLM
    """
    
    if not user_query or not user_query.strip():
        return {
            "status": "error",
            "message": "user_query parameter is required and cannot be empty",
            "llm_instruction": "Ask user to provide a search query."
        }
    
    try:
        print(f"🔍 Tool 1 called with query: {user_query}")
        
        # Execute the workflow with the user query
        result = execute_material_search_tool(user_query)
        
        print(f"📊 Tool 1 result status: {result.get('status')}")
        print(f"📊 Results count: {len(result.get('results', []))}")
        
        return result
        
    except Exception as e:
        print(f"❌ Tool 1 execution failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Tool execution failed: {str(e)}",
            "llm_instruction": "Tell user there was a technical error. Ask them to try again."
        }


def _format_selected_material_for_ui(selected_material: Dict[str, Any]) -> str:
    """
    Format the selected material object into a UI-parseable format
    
    Args:
        selected_material: Single material object from search results
        
    Returns:
        Formatted string that UI can parse and use to trigger material creation APIs
    """
    
    if not selected_material:
        return json.dumps({
            "status": "error",
            "message": "selected_material parameter is required",
            "llm_instruction": "Ask user to select a material first."
        })
    
    # Validate required fields
    required_fields = ['material_number', 'material_type', 'plant']
    missing_fields = [field for field in required_fields if not selected_material.get(field)]
    
    if missing_fields:
        return json.dumps({
            "status": "error",
            "message": f"Missing required fields: {', '.join(missing_fields)}",
            "llm_instruction": "The selected material is missing required information. Ask user to search again."
        })
    
    try:
        print(f"🎯 Tool 2 called with material: {selected_material.get('material_number')}")
        
        # Format the material for UI consumption
        formatted_response = format_material_for_ui(selected_material)
        
        print(f"✅ Tool 2 formatted response for UI parsing")
        
        return formatted_response
        
    except Exception as e:
        print(f"❌ Tool 2 execution failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"Formatting failed: {str(e)}",
            "llm_instruction": "Tell user there was an error formatting the material data."
        })


# =============================================================================
# FASTMCP DECORATED TOOLS
# =============================================================================

@mcp.tool()
def search_materials_for_reference(user_query: str) -> Dict[str, Any]:
    """
    Search for materials based on user query and return selectable results
    
    Args:
        user_query: User's search request (e.g., "Create with reference material type FERT, number CWUTEST70, plant I00X where request id was - NMTE20250918124421633")
        
    Returns:
        Dict with search results and instructions for LLM
    """
    return _search_materials_for_reference(user_query)


@mcp.tool()
def format_selected_material_for_ui(selected_material: Dict[str, Any]) -> str:
    """
    Format the selected material object into a UI-parseable format that triggers API calls
    
    Args:
        selected_material: Single material object from search results with keys like 'material_number', 'material_type', 'plant', etc.
        
    Returns:
        Formatted string that UI can parse and use to trigger material creation APIs
    """
    return _format_selected_material_for_ui(selected_material)




# =============================================================================
# MCP HTTP SERVER FOR DEPLOYMENT (Replace server startup section)
# =============================================================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import json
from typing import Dict, Any, List, Optional, Union

# Create FastAPI app with MCP protocol compliance
app = FastAPI(
    title="Material Search MCP Server",
    description="MCP-compliant HTTP server for material search tools",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MCP PROTOCOL MODELS (Exact specification compliance)
# =============================================================================

class MCPRequest(BaseModel):
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: Optional[Union[str, int]] = Field(default=None, description="Request ID")
    method: str = Field(description="MCP method name")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Method parameters")

class MCPError(BaseModel):
    code: int = Field(description="Error code")
    message: str = Field(description="Error message")
    data: Optional[Any] = Field(default=None, description="Additional error data")

class MCPResponse(BaseModel):
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: Optional[Union[str, int]] = Field(default=None, description="Request ID")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Success result")
    error: Optional[MCPError] = Field(default=None, description="Error details")

# =============================================================================
# MCP PROTOCOL IMPLEMENTATION
# =============================================================================

@app.get("/")
async def server_info():
    """Server information endpoint"""
    return {
        "name": "Material Search Server",
        "version": "1.0.0",
        "protocol": "Model Context Protocol",
        "transport": "HTTP",
        "capabilities": {
            "tools": {
                "listChanged": False
            }
        },
        "tools": [
            "search_materials_for_reference",
            "format_selected_material_for_ui"
        ],
        "endpoints": {
            "mcp": "/mcp",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2025-09-21T12:58:00Z"}

@app.post("/mcp")
async def mcp_protocol_handler(request: MCPRequest) -> MCPResponse:
    """
    Main MCP protocol handler - handles all MCP method calls
    This is the endpoint MCP clients will connect to
    """
    
    try:
        print(f"🔵 MCP Request: {request.method} (ID: {request.id})")
        
        if request.method == "initialize":
            # MCP initialization handshake
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "Material Search Server",
                        "version": "1.0.0"
                    }
                }
            )
        
        elif request.method == "tools/list":
            # Return available tools for MCP client discovery
            tools = [
                {
                    "name": "search_materials_for_reference",
                    "description": "Search for materials based on user query and return selectable results for material reference creation",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "user_query": {
                                "type": "string",
                                "description": "User's search request (e.g., 'Create with reference material type FERT, number CWUTEST70, plant I00X')"
                            }
                        },
                        "required": ["user_query"]
                    }
                },
                {
                    "name": "format_selected_material_for_ui",
                    "description": "Format the selected material object into a UI-parseable format that triggers API calls",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "selected_material": {
                                "type": "object",
                                "description": "Single material object from search results with material details",
                                "properties": {
                                    "material_number": {"type": "string"},
                                    "material_type": {"type": "string"},
                                    "plant": {"type": "string"},
                                    "request_id": {"type": "string"},
                                    "sales_org": {"type": "string"},
                                    "distribution_channel": {"type": "string"},
                                    "storage_location": {"type": "string"},
                                    "warehouse": {"type": "string"}
                                },
                                "required": ["material_number", "material_type", "plant"]
                            }
                        },
                        "required": ["selected_material"]
                    }
                }
            ]
            
            print(f"✅ Returning {len(tools)} tools to MCP client")
            
            return MCPResponse(
                id=request.id,
                result={"tools": tools}
            )
        
        elif request.method == "tools/call":
            # Handle tool execution calls
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            
            print(f"🔧 Executing tool: {tool_name}")
            print(f"📥 Arguments: {json.dumps(arguments, indent=2)}")
            
            if tool_name == "search_materials_for_reference":
                user_query = arguments.get("user_query", "")
                
                if not user_query:
                    return MCPResponse(
                        id=request.id,
                        error=MCPError(
                            code=-32602,
                            message="Invalid params: user_query is required"
                        )
                    )
                
                # Execute Tool 1
                result = _search_materials_for_reference(user_query)
                
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    }
                )
            
            elif tool_name == "format_selected_material_for_ui":
                selected_material = arguments.get("selected_material", {})
                
                if not selected_material:
                    return MCPResponse(
                        id=request.id,
                        error=MCPError(
                            code=-32602,
                            message="Invalid params: selected_material is required"
                        )
                    )
                
                # Execute Tool 2
                formatted_response = _format_selected_material_for_ui(selected_material)
                
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": formatted_response
                            }
                        ]
                    }
                )
            
            else:
                return MCPResponse(
                    id=request.id,
                    error=MCPError(
                        code=-32601,
                        message=f"Method not found: {tool_name}"
                    )
                )
        
        elif request.method == "ping":
            # Handle ping requests
            return MCPResponse(
                id=request.id,
                result={}
            )
        
        else:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=-32601,
                    message=f"Method not found: {request.method}"
                )
            )
    
    except Exception as e:
        print(f"❌ MCP Protocol Error: {str(e)}")
        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=-32603,
                message=f"Internal error: {str(e)}"
            )
        )

# =============================================================================
# ADDITIONAL HTTP ENDPOINTS (For testing and alternative access)
# =============================================================================

@app.post("/test/search")
async def test_search_tool(request: Dict[str, str]):
    """Test endpoint for Tool 1 (non-MCP)"""
    try:
        user_query = request.get("user_query", "")
        result = _search_materials_for_reference(user_query)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/format")  
async def test_format_tool(request: Dict[str, Any]):
    """Test endpoint for Tool 2 (non-MCP)"""
    try:
        selected_material = request.get("selected_material", {})
        result = _format_selected_material_for_ui(selected_material)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting MCP HTTP Server for Deployment")
    print("=" * 70)
    print("🏷️  Server: Material Search MCP Server")
    print("🌐 Protocol: Model Context Protocol over HTTP")
    print("🔧 Tools Available:")
    print("   1. search_materials_for_reference")
    print("   2. format_selected_material_for_ui")
    print("=" * 70)
    print("📡 MCP Endpoint: POST /mcp")
    print("💚 Health Check: GET /health")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🧪 Test Endpoints:")
    print("   - POST /test/search")
    print("   - POST /test/format")
    print("=" * 70)
    
    # Server configuration
    host = "0.0.0.0"  # Listen on all interfaces for deployment
    port = 8000
    
    print(f"🌐 Starting server on {host}:{port}")
    print("🔌 MCP clients can connect to: http://your-server:8000/mcp")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            access_log=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 MCP Server stopped by user")
    except Exception as e:
        print(f"❌ Server startup error: {e}")
