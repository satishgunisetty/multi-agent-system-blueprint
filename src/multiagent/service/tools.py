"""
Source-to-Pay Multi-Agent Tools - CORRECTED VERSION
Organized by functional domain and business process
"""

import json
from datetime import datetime, timedelta
from typing import List
from langchain_core.tools import tool
from src.multiagent.service.data_service import (
    data_store,
    InvoiceStatus,
    POStatus,
    ServiceNowTicket,
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def success_response(data: dict, message: str = "Success") -> str:
    """Return structured JSON for successful responses."""
    return json.dumps({"success": True, "message": message, "data": data}, default=str)


def error_response(message: str) -> str:
    """Return structured JSON for errors."""
    return json.dumps({"success": False, "error": message})


# ============================================================
# PURCHASE ORDER (PO) MANAGEMENT TOOLS
# ============================================================


@tool
def get_po_status(po_number: str) -> str:
    """Get purchase order status and details.

    Args:
        po_number: Purchase order number to look up.

    Returns:
        JSON string with structured PO details or error.
    """
    po = data_store.purchase_orders.get(po_number)
    if po:
        data = {
            "po_number": po.po_number,
            "vendor": po.vendor,
            "amount": po.amount,
            "status": po.status.value,
            "items": po.items,
            "created_date": po.created_date.strftime("%Y-%m-%d"),
        }
        return success_response(data, f"PO {po_number} retrieved successfully")
    return error_response(f"PO {po_number} not found")


@tool
def list_all_pos() -> str:
    """Return a list of all purchase orders in the system.

    Returns:
        JSON string with all POs and their basic details
    """
    pos = []
    for po in data_store.purchase_orders.values():
        pos.append(
            {
                "po_number": po.po_number,
                "vendor": po.vendor,
                "amount": po.amount,
                "status": po.status.value,
                "created_date": po.created_date.strftime("%Y-%m-%d"),
            }
        )
    return success_response({"purchase_orders": pos}, "All POs retrieved successfully")


@tool
def update_po_quantity(po_number: str, new_quantity: int) -> str:
    """Update the quantity of the first item in a purchase order.

    Args:
        po_number: Purchase order number to update.
        new_quantity: New quantity for the first item.

    Returns:
        JSON string with structured update result or error.
    """
    po = data_store.purchase_orders.get(po_number)
    if po and po.items:
        old_qty = po.items[0]["qty"]
        po.items[0]["qty"] = new_quantity
        po.amount = new_quantity * po.items[0]["price"]
        data = {
            "po_number": po_number,
            "old_quantity": old_qty,
            "new_quantity": new_quantity,
            "new_amount": po.amount,
        }
        return success_response(data, f"PO {po_number} updated successfully")
    return error_response(f"Could not update PO {po_number}")


@tool
def bulk_update_po_quantities(po_updates: List[dict]) -> str:
    """Update multiple purchase order quantities.

    Args:
        po_updates: List of dicts with {"po_number": str, "new_quantity": int}.

    Returns:
        JSON string with results for each PO.
    """
    results = []
    for update in po_updates:
        po_number = update.get("po_number")
        qty = update.get("new_quantity")
        res = json.loads(update_po_quantity(po_number, qty))
        results.append(res)
    return success_response({"results": results}, "Bulk PO update completed")


# ============================================================
# INVOICE MANAGEMENT TOOLS
# ============================================================


@tool
def get_invoice_status(invoice_number: str) -> str:
    """Get status of a specific invoice.

    Args:
        invoice_number: Invoice ID.

    Returns:
        JSON with invoice details or error.
    """
    inv = next(
        (i for i in data_store.invoices if i.invoice_number == invoice_number), None
    )
    if inv:
        data = {
            "invoice_number": inv.invoice_number,
            "po_number": inv.po_number,
            "vendor": inv.vendor,
            "amount": inv.amount,
            "status": inv.status.value,
            "rejection_reason": inv.rejection_reason,
        }
        return success_response(
            data, f"Invoice {invoice_number} retrieved successfully"
        )
    return error_response(f"Invoice {invoice_number} not found")


@tool
def list_all_invoices() -> str:
    """Return a list of all invoices in the system.

    Returns:
        JSON string with all invoices and their basic details
    """
    invoices = []
    for inv in data_store.invoices:
        invoices.append(
            {
                "invoice_number": inv.invoice_number,
                "po_number": inv.po_number,
                "vendor": inv.vendor,
                "amount": inv.amount,
                "status": inv.status.value,
                "rejection_reason": inv.rejection_reason,
            }
        )
    return success_response(
        {"invoices": invoices}, "All invoices retrieved successfully"
    )


@tool
def get_pending_invoices(days_back: int = 7) -> str:
    """Get invoices pending approval in the last N days.

    Args:
        days_back: Number of days to look back.

    Returns:
        JSON string with pending invoices.
    """
    pending = [
        {
            "invoice_number": inv.invoice_number,
            "po_number": inv.po_number,
            "vendor": inv.vendor,
            "amount": inv.amount,
            "status": inv.status.value,
        }
        for inv in data_store.invoices
        if inv.status == InvoiceStatus.PENDING
    ]
    return success_response(
        {"pending_invoices": pending}, f"Found {len(pending)} pending invoices"
    )


@tool
def get_rejected_invoices(days_back: int = 7) -> str:
    """Get invoices rejected in the last N days.

    Args:
        days_back: Number of days to look back.

    Returns:
        JSON string with list of rejected invoices.
    """
    rejected = [
        {
            "invoice_number": inv.invoice_number,
            "po_number": inv.po_number,
            "vendor": inv.vendor,
            "amount": inv.amount,
            "rejection_reason": inv.rejection_reason,
            "status": inv.status.value,
        }
        for inv in data_store.invoices
        if inv.status == InvoiceStatus.REJECTED
    ]
    return success_response(
        {"rejected_invoices": rejected}, f"Found {len(rejected)} rejected invoices"
    )


@tool
def get_invoice_due_date(invoice_number: str) -> str:
    """Get due date for a specific invoice (assume 30 days after PO created date).

    Args:
        invoice_number: Invoice ID.

    Returns:
        JSON with due date or error.
    """
    inv = next(
        (i for i in data_store.invoices if i.invoice_number == invoice_number), None
    )
    if inv:
        po = data_store.purchase_orders.get(inv.po_number)
        if not po:
            return error_response(
                f"PO {inv.po_number} not found for invoice {invoice_number}"
            )
        due_date = po.created_date + timedelta(days=30)
        data = {
            "invoice_number": inv.invoice_number,
            "po_number": inv.po_number,
            "due_date": due_date.strftime("%Y-%m-%d"),
        }
        return success_response(
            data, f"Due date for invoice {invoice_number} retrieved"
        )
    return error_response(f"Invoice {invoice_number} not found")


@tool
def get_invoice_rejection_reason(invoice_number: str) -> str:
    """Get rejection reason for a rejected invoice.

    Args:
        invoice_number: Invoice ID.

    Returns:
        JSON with rejection reason or error.
    """
    inv = next(
        (i for i in data_store.invoices if i.invoice_number == invoice_number), None
    )
    if not inv:
        return error_response(f"Invoice {invoice_number} not found")
    if inv.status != InvoiceStatus.REJECTED:
        return error_response(f"Invoice {invoice_number} is not rejected")
    data = {
        "invoice_number": inv.invoice_number,
        "po_number": inv.po_number,
        "vendor": inv.vendor,
        "amount": inv.amount,
        "rejection_reason": inv.rejection_reason,
    }
    return success_response(data, f"Rejection reason for {invoice_number} retrieved")


@tool
def handle_rejected_invoice(invoice_number: str) -> str:
    """Handle rejected invoice by fetching rejection reason and suggesting ticket creation. It won't create the ticket itself.
      You can use this method if someone needs assistance with a rejected invoice.

    Args:
        invoice_number: Invoice ID.

    Returns:
        JSON with rejection reason and suggested ticket details.
    """
    inv = next(
        (i for i in data_store.invoices if i.invoice_number == invoice_number), None
    )
    if not inv:
        return error_response(f"Invoice {invoice_number} not found")

    if inv.status != InvoiceStatus.REJECTED:
        return error_response(f"Invoice {invoice_number} is not rejected")

    # Prepare rejection details
    rejection_reason = inv.rejection_reason or "Unknown reason"

    ticket_suggestion = {
        "suggested_ticket": {
            "title": f"Rejected Invoice {invoice_number}",
            "description": f"Invoice {invoice_number} for PO {inv.po_number} (Vendor: {inv.vendor}) "
            f"was rejected. Reason: {rejection_reason}.",
            "priority": "High",
        }
    }

    data = {
        "invoice_number": inv.invoice_number,
        "po_number": inv.po_number,
        "vendor": inv.vendor,
        "amount": inv.amount,
        "rejection_reason": rejection_reason,
        **ticket_suggestion,
    }

    return success_response(
        data,
        f"Invoice {invoice_number} is rejected. Reason: {rejection_reason}. Suggest creating ServiceNow ticket.",
    )


# ============================================================
# PAYMENT MANAGEMENT TOOLS
# ============================================================


@tool
def get_due_payments(days_ahead: int = 30) -> str:
    """Get POs whose payments are expected within the next N days.

    Args:
        days_ahead: Look-ahead period in days.

    Returns:
        JSON string with upcoming payments.
    """
    upcoming = []
    today = datetime.now().date()
    for po in data_store.purchase_orders.values():
        if po.status in [POStatus.INVOICED, POStatus.CLOSED]:
            expected_date = (po.created_date + timedelta(days=30)).date()
            days_until_payment = (expected_date - today).days
            if 0 <= days_until_payment <= days_ahead:
                upcoming.append(
                    {
                        "po_number": po.po_number,
                        "vendor": po.vendor,
                        "amount": po.amount,
                        "expected_payment_date": expected_date.strftime("%Y-%m-%d"),
                    }
                )
    return success_response(
        {"upcoming_payments": upcoming}, f"Found {len(upcoming)} upcoming payments"
    )


@tool
def get_due_payment_for_po(po_number: str) -> str:
    """Get due payment details for a specific PO.

    Args:
        po_number: The purchase order number.

    Returns:
        JSON with due payment details for the PO.
    """
    invoices = [
        inv
        for inv in data_store.invoices
        if inv.po_number == po_number and inv.status == InvoiceStatus.APPROVED
    ]

    if not invoices:
        return error_response(f"No due payments found for PO {po_number}")

    data = []
    for inv in invoices:
        data.append(
            {
                "invoice_number": inv.invoice_number,
                "po_number": inv.po_number,
                "vendor": inv.vendor,
                "amount": inv.amount,
                "status": inv.status.value,
            }
        )

    return success_response(
        data, f"Found {len(invoices)} due payment(s) for PO {po_number}"
    )


# ============================================================
# VENDOR MANAGEMENT TOOLS
# ============================================================


@tool
def list_all_vendors() -> str:
    """Return a list of all vendors from POs and invoices.

    Returns:
        JSON string with unique vendor names
    """
    vendors = set()
    for po in data_store.purchase_orders.values():
        vendors.add(po.vendor)
    for inv in data_store.invoices:
        vendors.add(inv.vendor)
    return success_response(
        {"vendors": list(vendors)}, "All vendors retrieved successfully"
    )


@tool
def get_vendor_summary(vendor: str) -> str:
    """Get summary of POs and invoices for a vendor.

    Args:
        vendor: Vendor name.

    Returns:
        JSON string with counts and amounts.
    """
    pos = [po for po in data_store.purchase_orders.values() if po.vendor == vendor]
    invs = [inv for inv in data_store.invoices if inv.vendor == vendor]
    data = {
        "vendor": vendor,
        "total_pos": len(pos),
        "total_invoices": len(invs),
        "total_po_amount": sum(po.amount for po in pos),
        "total_invoice_amount": sum(inv.amount for inv in invs),
    }
    return success_response(data, f"Vendor summary for {vendor} retrieved")


# ============================================================
# SERVICENOW TICKET MANAGEMENT TOOLS
# ============================================================


@tool
def create_servicenow_ticket(
    entity_type: str,
    entity_id: str,
    title: str,
    description: str,
    priority: str = "Medium",
) -> str:
    """Check system records first; create a ServiceNow ticket only if entity is missing.

    Args:
        entity_type: Type of entity ('po', 'invoice', 'vendor', 'other').
        entity_id: ID of the entity (PO number, Invoice ID, Vendor ID). Ignored if 'other'.
        title: The ticket title.
        description: Detailed description of the issue.
        priority: Priority level (Low, Medium, High, Critical).

    Returns:
        JSON string with result (system check, existing ticket, or new ticket).
    """
    entity_type = entity_type.lower()

    # Check if entity exists in system
    if entity_type == "po":
        if entity_id in data_store.purchase_orders:
            return success_response(
                {"entity_type": entity_type, "entity_id": entity_id},
                f"PO {entity_id} exists in the system. No ticket needed.",
            )
    elif entity_type == "invoice":
        if entity_id in data_store.invoices:
            return success_response(
                {"entity_type": entity_type, "entity_id": entity_id},
                f"Invoice {entity_id} exists in the system. No ticket needed.",
            )
    elif entity_type == "vendor":
        if entity_id in data_store.vendors:
            return success_response(
                {"entity_type": entity_type, "entity_id": entity_id},
                f"Vendor {entity_id} exists in the system. No ticket needed.",
            )

    # Check if a ticket already exists for this entity
    if entity_type != "other":
        for ticket in data_store.servicenow_tickets:
            if entity_id and entity_id in ticket.description:
                data = {
                    "ticket_id": ticket.ticket_id,
                    "title": ticket.title,
                    "description": ticket.description,
                    "status": ticket.status.value,
                    "priority": ticket.priority,
                    "created_date": ticket.created_date.strftime("%Y-%m-%d %H:%M:%S"),
                }
                return success_response(
                    data,
                    f"A ServiceNow ticket already exists for {entity_type.upper()} {entity_id}: {ticket.ticket_id}",
                )

    # Create new ticket if entity missing and no existing ticket
    ticket_id = f"INC{len(data_store.servicenow_tickets) + 1000001}"
    full_desc = (
        f"[{entity_type.upper()} {entity_id}] {description}"
        if entity_type != "other"
        else description
    )
    ticket = ServiceNowTicket(
        ticket_id=ticket_id,
        title=title,
        description=full_desc,
        priority=priority,
    )
    data_store.servicenow_tickets.append(ticket)

    data = {
        "ticket_id": ticket.ticket_id,
        "title": ticket.title,
        "description": ticket.description,
        "status": ticket.status.value,
        "priority": ticket.priority,
        "created_date": ticket.created_date.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return success_response(
        data,
        f"ServiceNow ticket {ticket_id} created successfully for {entity_type.upper()} {entity_id if entity_id else ''}".strip(),
    )


@tool
def get_ticket_status(ticket_id: str) -> str:
    """Return status of a ServiceNow ticket.

    Args:
        ticket_id: ID of the ticket.

    Returns:
        JSON string with ticket details or error.
    """
    ticket = next(
        (t for t in data_store.servicenow_tickets if t.ticket_id == ticket_id), None
    )
    if ticket:
        data = {
            "ticket_id": ticket.ticket_id,
            "title": ticket.title,
            "status": ticket.status.value,
            "priority": ticket.priority,
            "created_date": ticket.created_date.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return success_response(data, f"Ticket {ticket_id} retrieved")
    return error_response(f"Ticket {ticket_id} not found")


@tool
def list_servicenow_tickets() -> str:
    """List all ServiceNow tickets with their details.

    Returns:
        JSON string containing a list of all ServiceNow tickets.
    """
    if not data_store.servicenow_tickets:
        return error_response("No ServiceNow tickets found")

    tickets = []
    for t in data_store.servicenow_tickets:
        tickets.append(
            {
                "ticket_id": t.ticket_id,
                "title": t.title,
                "status": t.status.value,
                "priority": t.priority,
                "created_date": t.created_date.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return success_response(
        {"servicenow_tickets": tickets}, "All ServiceNow tickets retrieved successfully"
    )


# ============================================================
# SESSION & ANALYTICS TOOLS
# ============================================================


@tool
def get_conversation_summary(session_id: str) -> str:
    """Return a summary of recent queries in a session.

    Args:
        session_id: Current session ID.

    Returns:
        JSON string with conversation summary.
    """
    queries = [
        q["query"] for q in data_store.user_queries if q["session_id"] == session_id
    ]
    data = {
        "session_id": session_id,
        "recent_queries": queries[-5:],
        "total_queries": len(queries),
        "summary": f"In this session, you made {len(queries)} queries covering POs, invoices, tickets, and updates.",
    }
    return success_response(data, "Conversation summary retrieved")


# ============================================================
# TOOL COLLECTIONS FOR AGENTS
# ============================================================

# SAP Agent Tools - Purchase Order and Invoice Management
sap_tools = [
    # Purchase Order Tools
    get_po_status,
    list_all_pos,
    update_po_quantity,
    bulk_update_po_quantities,
    # Invoice Management Tools
    get_invoice_status,
    list_all_invoices,
    get_pending_invoices,
    get_rejected_invoices,
    get_invoice_due_date,
    get_invoice_rejection_reason,
    handle_rejected_invoice,
    # Payment Management Tools
    get_due_payments,
    get_due_payment_for_po,
    # Vendor Management Tools
    list_all_vendors,
    get_vendor_summary,
]

# ServiceNow Agent Tools - Incident and Request Management
snow_tools = [
    handle_rejected_invoice,
    create_servicenow_ticket,
    get_ticket_status,
    list_servicenow_tickets,
]

# Supervisor Agent Tools - Session Management and Analytics
supervisor_tools = [
    get_conversation_summary,
]

# Combined tools for comprehensive agents
stp_tools = sap_tools + snow_tools + supervisor_tools

# Tool collections by business function
procurement_tools = [
    get_po_status,
    list_all_pos,
    update_po_quantity,
    bulk_update_po_quantities,
]

accounts_payable_tools = [
    get_invoice_status,
    list_all_invoices,
    get_pending_invoices,
    get_rejected_invoices,
    get_invoice_due_date,
    get_invoice_rejection_reason,
    handle_rejected_invoice,
    get_due_payments,
    get_due_payment_for_po,
]

vendor_management_tools = [
    list_all_vendors,
    get_vendor_summary,
]

incident_management_tools = [
    create_servicenow_ticket,
    get_ticket_status,
    list_servicenow_tickets,
]
