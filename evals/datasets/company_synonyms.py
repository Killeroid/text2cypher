"""The company graph, asked in words that are not the names in its schema.

The questions of ``evals/datasets/company.py`` mostly use the words of the schema, so the lexical
filter does well on them. These are the same graph asked as a person would: "manager" for
``REPORTS_TO``, "clients" for ``Customer``, "staff" for ``Employee``. Each question has the
query that answers it, and the lexical filter's note (``whole``, or how much it kept and which
names the gold query needed and it dropped) says where it falls short. They measure an LLM
schema filter against it. Like the company graph, the rows mean nothing, so a final query is
judged by passing validation.
"""

from evals.dataset import Dataset, Question
from evals.datasets.company import COMPANY

QUESTIONS = (
    Question(
        "s01",
        "synonym of a relationship type",
        "Who is the manager of Sam Okoye?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:REPORTS_TO]->(m:Employee) RETURN m.name",
    ),
    Question(
        "s02",
        "synonym of a label, no schema word",
        "Which clients have a balance above 1000?",
        "MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account) WHERE a.balance > 1000 RETURN c.name",
    ),
    Question(
        "s03",
        "words that match, a control",
        "In which cities do our staff sit?",
        "MATCH (:Employee)-[:SITS_IN]->(:Office)-[:LOCATED_IN]->(c:City) RETURN DISTINCT c.name",
    ),
    Question(
        "s04",
        "no schema word",
        "What did Sam Okoye spend his time on?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:ASSIGNED_TO]->(t:Task) RETURN t.title",
    ),
    Question(
        "s05",
        "words that match, a control",
        "Which people at the company hold a certificate called Cloud?",
        "MATCH (e:Employee)-[:HOLDS]->(:Certification {name: 'Cloud'}) RETURN e.name",
    ),
    Question(
        "s06",
        "a label and a type with other words",
        "Which suppliers provide the goods stored in the Rotterdam depot?",
        "MATCH (:Warehouse {name: 'Rotterdam'})-[:STOCKS]->(:Product)-[:SUPPLIED_BY]->(s:Supplier) RETURN DISTINCT s.name",
    ),
    Question(
        "s07",
        "a type with another word",
        "Which staff members have a laptop registered to them?",
        "MATCH (e:Employee)-[:OWNS]->(:Asset)<-[:REGISTERED_AS]-(:Device {model: 'laptop'}) RETURN e.name",
    ),
    Question(
        "s08",
        "words that match, a control",
        "Who answers the tickets that clients raised?",
        "MATCH (:Customer)-[:RAISED]->(:Ticket)-[:ASSIGNED_TO]->(e:Employee) RETURN DISTINCT e.name",
    ),
    Question(
        "s09",
        "a label and a type with other words",
        "Which workers took part in discussions about the budget document?",
        "MATCH (e:Employee)-[:ATTENDED]->(:Meeting)-[:ABOUT]->(:Document {title: 'Budget'}) RETURN e.name",
    ),
    Question(
        "s10",
        "a long chain, other words",
        "How much did we get paid for the invoices of Acme?",
        "MATCH (:Customer {name: 'Acme'})-[:PLACED]->(:Order)-[:BILLED_BY]->(:Invoice)-[:SETTLED_BY]->(p:Payment) RETURN sum(p.amount)",
    ),
    Question(
        "s11",
        "a chain, other words",
        "Which developers wrote changes in the code stores used by the Phoenix project?",
        "MATCH (e:Employee)-[:AUTHORED]->(:Commit)<-[:HAS_COMMIT]-(:Repository)<-[:USES_REPOSITORY]-(:Project {name: 'Phoenix'}) RETURN DISTINCT e.name",
    ),
    Question(
        "s12",
        "a chain, other words",
        "Which goods are in the purchase history of Acme?",
        "MATCH (:Customer {name: 'Acme'})-[:PLACED]->(:Order)-[:CONTAINS]->(:OrderLine)-[:OF_PRODUCT]->(p:Product) RETURN DISTINCT p.name",
    ),
    Question(
        "s13",
        "a shared neighbour, other words",
        "Who shares a workplace with Sam Okoye?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:SITS_IN]->(:Office)<-[:SITS_IN]-(c:Employee) RETURN c.name",
    ),
    Question(
        "s14",
        "a label with another word",
        "Which countries are our clients based in?",
        "MATCH (:Customer)-[:LOCATED_IN]->(c:Country) RETURN DISTINCT c.name",
    ),
    Question(
        "s15",
        "two relationships, other words",
        "Which promotions brought in prospects that became customers?",
        "MATCH (l:Lead)-[:CONVERTED_TO]->(:Customer) MATCH (l)-[:FROM_CAMPAIGN]->(c:Campaign) RETURN DISTINCT c.name",
    ),
    Question(
        "s16",
        "a relationship type with another word",
        "Which people have the same boss as Sam Okoye?",
        "MATCH (:Employee {name: 'Sam Okoye'})-[:REPORTS_TO]->(b:Employee)<-[:REPORTS_TO]-(e:Employee) RETURN e.name",
    ),
    Question(
        "s17",
        "two labels, other words",
        "Which technicians hold a Cloud credential and know Python?",
        "MATCH (e:Employee)-[:HOLDS]->(:Certification {name: 'Cloud'}) MATCH (e)-[:HAS_SKILL]->(:Skill {name: 'Python'}) RETURN e.name",
    ),
    Question(
        "s18",
        "a long chain, other words",
        "Which problems lay behind the requests for help that Acme sent in?",
        "MATCH (:Customer {name: 'Acme'})-[:RAISED]->(:Ticket)-[:CAUSED_BY]->(i:Incident) RETURN DISTINCT i.title",
    ),
    Question(
        "s19",
        "a relationship type with another word",
        "Who leads the work done for Acme?",
        "MATCH (e:Employee)-[:MANAGES]->(:Project)-[:FOR_CUSTOMER]->(:Customer {name: 'Acme'}) RETURN DISTINCT e.name",
    ),
    Question(
        "s20",
        "a relationship type with another word",
        "Which departments finance projects that are active?",
        "MATCH (d:Department)-[:FUNDS]->(p:Project {status: 'active'}) RETURN DISTINCT d.name",
    ),
)

COMPANY_SYNONYMS = Dataset(
    COMPANY.graph, COMPANY.data, QUESTIONS, (), compare_rows=COMPANY.compare_rows
)
