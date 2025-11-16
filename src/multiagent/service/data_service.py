from enum import Enum
from typing import List, Dict
from datetime import datetime, timedelta


# -----------------------------
# Enums for statuses
# -----------------------------
class POStatus(Enum):
    OPEN = "Open"
    RELEASED = "Released"
    RECEIVED = "Received"
    INVOICED = "Invoiced"
    CLOSED = "Closed"


class InvoiceStatus(Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    PAID = "Paid"


class TicketStatus(Enum):
    NEW = "New"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    CLOSED = "Closed"


# -----------------------------
# Models
# -----------------------------
class PurchaseOrder:
    def __init__(
        self,
        po_number: str,
        vendor: str,
        amount: float,
        status: POStatus,
        items: List[Dict],
        created_date: datetime,
    ):
        self.po_number = po_number
        self.vendor = vendor
        self.amount = amount
        self.status = status
        self.items = items
        self.created_date = created_date


class Invoice:
    def __init__(
        self,
        invoice_number: str,
        po_number: str,
        vendor: str,
        amount: float,
        status: InvoiceStatus,
        rejection_reason: str = None,
    ):
        self.invoice_number = invoice_number
        self.po_number = po_number
        self.vendor = vendor
        self.amount = amount
        self.status = status
        self.rejection_reason = rejection_reason


class ServiceNowTicket:
    def __init__(
        self,
        ticket_id: str,
        title: str,
        description: str,
        priority: str = "Medium",
        status: TicketStatus = TicketStatus.NEW,
    ):
        self.ticket_id = ticket_id
        self.title = title
        self.description = description
        self.priority = priority
        self.status = status
        self.created_date = datetime.now()


# -----------------------------
# Fixed Mock Data Store
# -----------------------------
class MockDataStore:
    def __init__(self):
        self.vendors = self._create_vendors()
        self.purchase_orders = self._create_mock_pos()
        self.invoices = self._create_mock_invoices()
        self.servicenow_tickets: List[ServiceNowTicket] = self._create_mock_tickets()
        self.user_queries: List[Dict] = self._create_mock_user_queries()
        self.tool_call_logs: List[Dict] = []

    # -----------------------------
    # Vendors
    # -----------------------------
    def _create_vendors(self) -> List[str]:
        return [
            "Acme Corp",
            "Tech Solutions Ltd",
            "Office Supplies Inc",
            "Global Electronics",
            "Alpha Traders",
            "Beta Distributors",
            "Gamma Wholesale",
            "Delta Imports",
            "Sigma Retailers",
            "Omega Supplies",
        ]

    # -----------------------------
    # Purchase Orders
    # -----------------------------
    def _create_mock_pos(self) -> Dict[str, PurchaseOrder]:
        pos = {}
        for i in range(10):
            po_number = f"PO{1000+i}"
            vendor = self.vendors[i]
            amount = (i + 1) * 5000
            status = list(POStatus)[i % len(POStatus)]
            items = [
                {"item": f"Product-{i+1}", "qty": (i + 1) * 2, "price": 250.0},
                {"item": f"Accessory-{i+1}", "qty": 1, "price": 500.0},
            ]
            pos[po_number] = PurchaseOrder(
                po_number=po_number,
                vendor=vendor,
                amount=amount,
                status=status,
                items=items,
                created_date=datetime.now() - timedelta(days=i * 3),
            )
        return pos

    # -----------------------------
    # Invoices
    # -----------------------------
    def _create_mock_invoices(self) -> List[Invoice]:
        invoices = []
        for i, po_number in enumerate(self.purchase_orders.keys()):
            vendor = self.purchase_orders[po_number].vendor
            amount = self.purchase_orders[po_number].amount
            status = list(InvoiceStatus)[i % len(InvoiceStatus)]
            rejection_reason = None
            if status == InvoiceStatus.REJECTED:
                rejection_reason = "PO amount mismatch"
            invoices.append(
                Invoice(
                    invoice_number=f"INV-{2000+i}",
                    po_number=po_number,
                    vendor=vendor,
                    amount=amount,
                    status=status,
                    rejection_reason=rejection_reason,
                )
            )
        return invoices

    # -----------------------------
    # ServiceNow Tickets
    # -----------------------------
    def _create_mock_tickets(self) -> List[ServiceNowTicket]:
        tickets = []
        for i in range(10):
            tickets.append(
                ServiceNowTicket(
                    ticket_id=f"INC{1000000+i}",
                    title=f"Issue {i+1}",
                    description=f"Mock issue description for {list(self.purchase_orders.keys())[i]}",
                    priority=["Low", "Medium", "High", "Critical"][i % 4],
                    status=list(TicketStatus)[i % len(TicketStatus)],
                )
            )
        return tickets

    # -----------------------------
    # User queries for conversation summary
    # -----------------------------
    def _create_mock_user_queries(self) -> List[Dict]:
        return [
            {
                "session_id": f"session-{i%3+1}",
                "timestamp": datetime.now() - timedelta(days=i),
                "query": f"Query {i+1}",
            }
            for i in range(10)
        ]


# -----------------------------
# Initialize global data store
# -----------------------------
data_store = MockDataStore()
