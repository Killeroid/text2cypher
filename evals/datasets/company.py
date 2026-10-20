"""A large schema for measuring the schema filter without a model or a database.

The film graph of ``evals/datasets/movies.py`` has six labels, too few to say how much the filter
prunes or whether it keeps what a question needs. This is a company's graph with 36 labels
and 47 relationship types, written as a ``Snapshot`` by hand (no data is needed to read a
schema), and questions with the query that answers each. ``measure`` runs the filter on every
question and reports how much it kept and which names the gold query needed and it dropped;
``check_gold`` validates every gold query against the schema, offline.
"""

from cypher_guardrail import AsyncPropertiesValidator, AsyncSchemaValidator, Snapshot

from evals.dataset import Dataset, Question, Seeded
from evals.judge import filter_note
from text2cypher import GraphSchema
from text2cypher.schema import schema_from_snapshot

S, I, F = "String", "Integer", "Float"

NODES: dict[str, dict[str, str]] = {
    "Employee": {"name": S, "email": S, "hired": I, "salary": F},
    "Department": {"name": S, "budget": F},
    "Team": {"name": S},
    "Role": {"title": S, "level": I},
    "Skill": {"name": S},
    "Certification": {"name": S, "year": I},
    "Office": {"name": S, "floor": I},
    "City": {"name": S, "population": I},
    "Country": {"name": S, "code": S},
    "Region": {"name": S},
    "Project": {"name": S, "status": S, "deadline": I},
    "Task": {"title": S, "status": S, "estimate": F},
    "Milestone": {"name": S, "due": I},
    "Customer": {"name": S, "segment": S},
    "Account": {"number": S, "balance": F},
    "Contact": {"name": S, "phone": S},
    "Lead": {"name": S, "score": I},
    "Campaign": {"name": S, "spend": F},
    "Order": {"number": S, "total": F, "placed": I},
    "OrderLine": {"quantity": I, "price": F},
    "Product": {"name": S, "price": F, "sku": S},
    "Category": {"name": S},
    "Supplier": {"name": S, "rating": F},
    "Warehouse": {"name": S, "capacity": I},
    "Invoice": {"number": S, "amount": F, "due": I},
    "Payment": {"amount": F, "paid": I, "method": S},
    "Contract": {"title": S, "value": F, "signed": I},
    "Ticket": {"title": S, "priority": S, "opened": I},
    "Incident": {"title": S, "severity": I},
    "Asset": {"tag": S, "cost": F},
    "Device": {"serial": S, "model": S},
    "Repository": {"name": S, "language": S},
    "Commit": {"hash": S, "message": S, "date": I},
    "Release": {"version": S, "date": I},
    "Meeting": {"title": S, "date": I},
    "Document": {"title": S, "pages": I},
}

RELATIONSHIPS: dict[tuple[str, str, str], dict[str, str]] = {
    ("Employee", "WORKS_IN", "Department"): {"since": I},
    ("Employee", "MEMBER_OF", "Team"): {},
    ("Employee", "REPORTS_TO", "Employee"): {},
    ("Employee", "HAS_ROLE", "Role"): {},
    ("Employee", "HAS_SKILL", "Skill"): {"level": I},
    ("Employee", "HOLDS", "Certification"): {},
    ("Employee", "SITS_IN", "Office"): {},
    ("Employee", "ASSIGNED_TO", "Task"): {"hours": F},
    ("Employee", "MANAGES", "Project"): {},
    ("Employee", "AUTHORED", "Commit"): {},
    ("Employee", "ATTENDED", "Meeting"): {},
    ("Employee", "OWNS", "Asset"): {},
    ("Department", "PART_OF", "Department"): {},
    ("Department", "FUNDS", "Project"): {"amount": F},
    ("Team", "BELONGS_TO", "Department"): {},
    ("Office", "LOCATED_IN", "City"): {},
    ("City", "IN_COUNTRY", "Country"): {},
    ("Country", "IN_REGION", "Region"): {},
    ("Project", "HAS_TASK", "Task"): {},
    ("Project", "HAS_MILESTONE", "Milestone"): {},
    ("Project", "FOR_CUSTOMER", "Customer"): {},
    ("Project", "USES_REPOSITORY", "Repository"): {},
    ("Task", "DEPENDS_ON", "Task"): {},
    ("Repository", "HAS_COMMIT", "Commit"): {},
    ("Repository", "HAS_RELEASE", "Release"): {},
    ("Customer", "HAS_ACCOUNT", "Account"): {},
    ("Customer", "HAS_CONTACT", "Contact"): {"primary": S},
    ("Customer", "PLACED", "Order"): {},
    ("Customer", "SIGNED", "Contract"): {},
    ("Customer", "RAISED", "Ticket"): {},
    ("Customer", "LOCATED_IN", "Country"): {},
    ("Lead", "CONVERTED_TO", "Customer"): {},
    ("Lead", "FROM_CAMPAIGN", "Campaign"): {},
    ("Order", "CONTAINS", "OrderLine"): {},
    ("OrderLine", "OF_PRODUCT", "Product"): {},
    ("Order", "BILLED_BY", "Invoice"): {},
    ("Invoice", "SETTLED_BY", "Payment"): {},
    ("Order", "SHIPPED_FROM", "Warehouse"): {"date": I},
    ("Product", "IN_CATEGORY", "Category"): {},
    ("Product", "SUPPLIED_BY", "Supplier"): {"cost": F},
    ("Category", "SUBCATEGORY_OF", "Category"): {},
    ("Warehouse", "STOCKS", "Product"): {"quantity": I},
    ("Ticket", "ASSIGNED_TO", "Employee"): {},
    ("Ticket", "CAUSED_BY", "Incident"): {},
    ("Device", "REGISTERED_AS", "Asset"): {},
    ("Meeting", "ABOUT", "Document"): {},
}


QUESTIONS = (
    Question(
        "l01",
        "label and property",
        "Which employees were hired after 2020?",
        "MATCH (e:Employee) WHERE e.hired > 2020 RETURN e.name",
    ),
    Question(
        "l02",
        "relationship word",
        "Who reports to Maria Lopez?",
        "MATCH (e:Employee)-[:REPORTS_TO]->(:Employee {name: 'Maria Lopez'}) RETURN e.name",
    ),
    Question(
        "l03",
        "two labels",
        "Which department does the employee Sam Okoye work in?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:WORKS_IN]->(d:Department) RETURN d.name",
    ),
    Question(
        "l04",
        "three hops",
        "Which countries have offices where employees from the Data team sit?",
        "MATCH (:Team {name: 'Data'})<-[:MEMBER_OF]-(:Employee)-[:SITS_IN]->(:Office)-[:LOCATED_IN]->(:City)-[:IN_COUNTRY]->(c:Country) RETURN DISTINCT c.name",
    ),
    Question(
        "l05",
        "aggregation",
        "How many tasks does each project have?",
        "MATCH (p:Project)-[:HAS_TASK]->(t:Task) RETURN p.name, count(t)",
    ),
    Question(
        "l06",
        "relationship property",
        "How many hours is Sam Okoye assigned to each task?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[a:ASSIGNED_TO]->(t:Task) RETURN t.title, a.hours",
    ),
    Question(
        "l07",
        "shared relationship type",
        "Which customers are located in Canada?",
        "MATCH (c:Customer)-[:LOCATED_IN]->(:Country {name: 'Canada'}) RETURN c.name",
    ),
    Question(
        "l08",
        "self relationship",
        "Which tasks does the task 'Design API' depend on?",
        "MATCH (:Task {title: 'Design API'})-[:DEPENDS_ON]->(t:Task) RETURN t.title",
    ),
    Question(
        "l09",
        "long chain",
        "What is the total value of the orders placed by Acme, by product category?",
        "MATCH (:Customer {name: 'Acme'})-[:PLACED]->(:Order)-[:CONTAINS]->(l:OrderLine)-[:OF_PRODUCT]->(:Product)-[:IN_CATEGORY]->(c:Category) RETURN c.name, sum(l.price * l.quantity)",
    ),
    Question(
        "l10",
        "synonym of a label",
        "Which staff members know Python?",
        "MATCH (e:Employee)-[:HAS_SKILL]->(:Skill {name: 'Python'}) RETURN e.name",
    ),
    Question(
        "l11",
        "synonym of a relationship type",
        "Who is the boss of Sam Okoye?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:REPORTS_TO]->(b:Employee) RETURN b.name",
    ),
    Question(
        "l12",
        "plural and tense",
        "Which projects did Maria Lopez manage?",
        "MATCH (:Employee {name: 'Maria Lopez'})-[:MANAGES]->(p:Project) RETURN p.name",
    ),
    Question(
        "l13",
        "no schema word",
        "What is the most expensive thing we sell?",
        "MATCH (p:Product) RETURN p.name ORDER BY p.price DESC LIMIT 1",
    ),
    Question(
        "l14",
        "property word only",
        "Which suppliers have a rating above 4?",
        "MATCH (s:Supplier) WHERE s.rating > 4 RETURN s.name",
    ),
    Question(
        "l15",
        "label and unrelated type word",
        "Which invoices were settled by a payment made by card?",
        "MATCH (i:Invoice)-[:SETTLED_BY]->(p:Payment {method: 'card'}) RETURN i.number",
    ),
    Question(
        "l16",
        "common words in names",
        "Which employees are members of a team in the Sales department?",
        "MATCH (e:Employee)-[:MEMBER_OF]->(:Team)-[:BELONGS_TO]->(:Department {name: 'Sales'}) RETURN e.name",
    ),
)


def _properties_by_type() -> dict[str, dict[str, str]]:
    """The properties of each relationship type; a type used between several labels has one set."""
    by_type: dict[str, dict[str, str]] = {}
    for (_, kind, _), properties in RELATIONSHIPS.items():
        by_type.setdefault(kind, {}).update(properties)
    return by_type


def snapshot() -> Snapshot:
    """The company graph as a snapshot, with the value types of every property."""
    by_type = _properties_by_type()
    return Snapshot.from_schema(
        nodes={label: list(properties) for label, properties in NODES.items()},
        relationships=list(RELATIONSHIPS),
        relationship_properties={kind: list(props) for kind, props in by_type.items()},
        node_property_types={
            label: {name: [type_] for name, type_ in properties.items()}
            for label, properties in NODES.items()
        },
        relationship_property_types={
            kind: {name: [type_] for name, type_ in props.items()}
            for kind, props in by_type.items()
        },
        graph="company",
    )


def schema() -> GraphSchema:
    return schema_from_snapshot(snapshot())


async def check_gold(question: Question) -> list[str]:
    """What the schema and properties validators object to in the gold query (offline)."""
    source = snapshot()
    _, schema_issues = await AsyncSchemaValidator(source).validate(question.gold)
    _, property_issues = await AsyncPropertiesValidator(source).validate(question.gold)
    return [issue["description"] for issue in [*schema_issues, *property_issues]]


def measure() -> list[tuple[Question, str]]:
    """Each question with the filter's note: how much it kept, and what it dropped that was needed."""
    full = schema()
    return [(question, filter_note(full, question)) for question in QUESTIONS]


_DUMMY = {S: "'x'", I: "1", F: "1.0"}


def _node(label: str) -> str:
    properties = ", ".join(
        f"{name}: {_DUMMY[type_]}" for name, type_ in NODES[label].items()
    )
    return f"({label.lower()}:{label} {{{properties}}})"


def _edge(source: str, kind: str, target: str, properties: dict[str, str]) -> str:
    inside = ", ".join(f"{name}: {_DUMMY[type_]}" for name, type_ in properties.items())
    return f"({source.lower()})-[:{kind} {{{inside}}}]->({target.lower()})"


# One node per label and one edge per path, so the graph has the shape of the schema and
# nothing more: a query can be validated and run on it, but its rows mean nothing
DATA = "CREATE " + ",\n ".join(
    [_node(label) for label in NODES]
    + [_edge(*path, properties) for path, properties in RELATIONSHIPS.items()]
)

SEEDED = (
    Seeded(
        "c01",
        "wrong direction",
        "Which department does the employee Sam Okoye work in?",
        "MATCH (:Employee {name: 'Sam Okoye'})<-[:WORKS_IN]-(d:Department) RETURN d.name",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:WORKS_IN]->(d:Department) RETURN d.name",
    ),
    Seeded(
        "c02",
        "relationship type, close name",
        "Which projects did Maria Lopez manage?",
        "MATCH (:Employee {name: 'Maria Lopez'})-[:MANAGED]->(p:Project) RETURN p.name",
        "MATCH (:Employee {name: 'Maria Lopez'})-[:MANAGES]->(p:Project) RETURN p.name",
    ),
    Seeded(
        "c03",
        "relationship type, other word",
        "Who is the boss of Sam Okoye?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:BOSS_OF]->(b:Employee) RETURN b.name",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:REPORTS_TO]->(b:Employee) RETURN b.name",
    ),
    Seeded(
        "c04",
        "label, typo",
        "Which employees were hired after 2020?",
        "MATCH (e:Employe) WHERE e.hired > 2020 RETURN e.name",
        "MATCH (e:Employee) WHERE e.hired > 2020 RETURN e.name",
    ),
    Seeded(
        "c05",
        "label, other word",
        "Which customers placed orders?",
        "MATCH (c:Client)-[:PLACED]->(o:Order) RETURN DISTINCT c.name",
        "MATCH (c:Customer)-[:PLACED]->(o:Order) RETURN DISTINCT c.name",
    ),
    Seeded(
        "c06",
        "property of a similar label",
        "What is the quantity of each order line?",
        "MATCH (l:Order) RETURN l.quantity",
        "MATCH (l:OrderLine) RETURN l.quantity",
    ),
    Seeded(
        "c07",
        "property of another label",
        "What is the salary of Sam Okoye?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:WORKS_IN]->(d:Department) RETURN d.salary",
        "MATCH (e:Employee {name: 'Sam Okoye'}) RETURN e.salary",
    ),
    Seeded(
        "c08",
        "property, close name",
        "Which suppliers have a rating above 4?",
        "MATCH (s:Supplier) WHERE s.rate > 4 RETURN s.name",
        "MATCH (s:Supplier) WHERE s.rating > 4 RETURN s.name",
    ),
    Seeded(
        "c09",
        "relationship property read from the node",
        "How many hours is Sam Okoye assigned to each task?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:ASSIGNED_TO]->(t:Task) RETURN t.title, t.hours",
        "MATCH (:Employee {name: 'Sam Okoye'})-[a:ASSIGNED_TO]->(t:Task) RETURN t.title, a.hours",
    ),
    Seeded(
        "c10",
        "unknown relationship type with a property",
        "How many hours is Sam Okoye assigned to each task?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[a:ASSIGNED]->(t:Task) RETURN t.title, a.hours",
        "MATCH (:Employee {name: 'Sam Okoye'})-[a:ASSIGNED_TO]->(t:Task) RETURN t.title, a.hours",
    ),
    Seeded(
        "c11",
        "parameter",
        "Which projects did Maria Lopez manage?",
        "MATCH (:Employee {name: $name})-[:MANAGES]->(p:Project) RETURN p.name",
        "MATCH (:Employee {name: 'Maria Lopez'})-[:MANAGES]->(p:Project) RETURN p.name",
    ),
    Seeded(
        "c12",
        "write",
        "Create a new task called Test.",
        "CREATE (:Task {title: 'Test'})",
        None,
    ),
    Seeded(
        "c13",
        "syntax",
        "Which suppliers have a rating above 4?",
        "MATCH (s:Supplier WHERE s.rating > 4 RETURN s.name",
        "MATCH (s:Supplier) WHERE s.rating > 4 RETURN s.name",
    ),
    Seeded(
        "c14",
        "two mistakes",
        "Which customers are located in Canada?",
        "MATCH (c:Customer)-[:IN]->(co:Country {title: 'Canada'}) RETURN c.name",
        "MATCH (c:Customer)-[:LOCATED_IN]->(:Country {name: 'Canada'}) RETURN c.name",
    ),
    Seeded(
        "c15",
        "path that does not exist, shared type",
        "Which offices are located in Canada?",
        "MATCH (o:Office)-[:LOCATED_IN]->(:Country {name: 'Canada'}) RETURN o.name",
        "MATCH (o:Office)-[:LOCATED_IN]->(:City)-[:IN_COUNTRY]->(:Country {name: 'Canada'}) RETURN o.name",
    ),
    Seeded(
        "c16",
        "path that does not exist, a hop missing",
        "Which employees attended meetings about the Q3 plan?",
        "MATCH (e:Employee)-[:ATTENDED]->(:Document {title: 'Q3 plan'}) RETURN e.name",
        "MATCH (e:Employee)-[:ATTENDED]->(:Meeting)-[:ABOUT]->(:Document {title: 'Q3 plan'}) RETURN e.name",
    ),
)

# The rows of this graph mean nothing, so a final query is judged by passing validation
COMPANY = Dataset("company", DATA, QUESTIONS, SEEDED, compare_rows=False)
