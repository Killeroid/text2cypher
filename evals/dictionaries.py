"""Data dictionaries for the trial graphs (ADR-054), for the format measurement (ADR-055).

They describe what the graphs already are, as the owner of such a graph would write it down:
plain business definitions, with the usual other name for a thing where there is one ("client",
"line manager"). They are not written to answer the questions, but a dictionary that says what a
relationship means will help with a question that uses other words for it, and that is what
``company_synonyms`` asks. Read a gain on that dataset as the most a dictionary can give there.
"""

from text2cypher import DataDictionary

MOVIES_DICTIONARY = DataDictionary(
    {
        "nodes": {
            "Person": {
                "description": "A person in the film industry: an actor, a director, or both.",
                "properties": {
                    "name": "Full name as credited.",
                    "born": "Year of birth.",
                },
            },
            "Movie": {
                "description": "A feature film.",
                "properties": {
                    "title": "Title as released.",
                    "year": "Year of first release.",
                    "rating": "Average critic score from 0 to 10.",
                },
            },
            "Genre": {
                "description": "A category of film, such as SciFi or Action.",
                "properties": {"name": "Name of the genre."},
            },
            "Country": {
                "description": "A country where people were born.",
                "properties": {"name": "Name of the country."},
            },
            "Studio": {
                "description": "A company that produces films.",
                "properties": {"name": "Name of the studio."},
            },
            "User": {
                "description": "A member of the review community who rates movies and follows other users.",
                "properties": {"name": "Display name."},
            },
        },
        "relationships": {
            "ACTED_IN": {
                "description": "A person appeared in a movie as a character.",
                "properties": {"role": "The character the person played."},
            },
            "DIRECTED": {"description": "A person directed a movie."},
            "IN_GENRE": {"description": "A movie belongs to a genre."},
            "KNOWS": {"description": "One person knows another."},
            "BORN_IN": {"description": "A person was born in a country."},
            "PRODUCED_BY": {"description": "A movie was produced by a studio."},
            "REVIEWED": {
                "description": "A user reviewed a movie.",
                "properties": {"stars": "The user's score, from 1 to 5."},
            },
            "FOLLOWS": {"description": "A user follows another user."},
        },
    }
)

COMPANY_DICTIONARY = DataDictionary(
    {
        "nodes": {
            "Employee": {
                "description": "A person employed by the company; also called staff or a worker.",
                "properties": {
                    "name": "Full name.",
                    "email": "Work email address.",
                    "hired": "Year the person was hired.",
                    "salary": "Annual salary.",
                },
            },
            "Department": {
                "description": "An organisational unit of the company, such as Finance or Engineering.",
                "properties": {
                    "name": "Name of the department.",
                    "budget": "Annual budget.",
                },
            },
            "Team": {
                "description": "A working group of employees inside a department."
            },
            "Role": {
                "description": "A job position that an employee can hold.",
                "properties": {
                    "title": "Name of the position.",
                    "level": "Seniority; a higher number is more senior.",
                },
            },
            "Skill": {
                "description": "A technical or professional ability, such as Python."
            },
            "Certification": {
                "description": "A professional credential that an employee can earn; also called a certificate.",
                "properties": {"year": "Year it was awarded."},
            },
            "Office": {
                "description": "A building where employees sit.",
                "properties": {"floor": "The floor of the building the office is on."},
            },
            "City": {"properties": {"population": "Number of inhabitants."}},
            "Country": {"properties": {"code": "Two-letter ISO country code."}},
            "Region": {
                "description": "A geographic area that groups countries, such as EMEA."
            },
            "Project": {
                "description": "A piece of work, usually delivered for a customer.",
                "properties": {
                    "status": "Where it stands now, such as 'active' or 'closed'.",
                    "deadline": "Year it is due.",
                },
            },
            "Task": {
                "description": "A unit of work within a project.",
                "properties": {
                    "status": "Where it stands now.",
                    "estimate": "Estimated effort in hours.",
                },
            },
            "Milestone": {
                "description": "A checkpoint that a project has to reach.",
                "properties": {"due": "Year it is due."},
            },
            "Customer": {
                "description": "An organisation the company sells to; also called a client.",
                "properties": {"segment": "Market segment the customer belongs to."},
            },
            "Account": {
                "description": "A customer's financial account with the company.",
                "properties": {"balance": "Money currently on the account."},
            },
            "Contact": {"description": "A person at a customer who can be reached."},
            "Lead": {
                "description": "A prospective customer that has not bought yet; also called a prospect.",
                "properties": {
                    "score": "How likely the lead is to convert, from 0 to 100."
                },
            },
            "Campaign": {
                "description": "A marketing promotion that brings in leads.",
                "properties": {"spend": "Money spent on the campaign."},
            },
            "Order": {
                "description": "A purchase placed by a customer.",
                "properties": {
                    "total": "Value of the whole order.",
                    "placed": "Year it was placed.",
                },
            },
            "OrderLine": {
                "description": "One product, and how many of it, within an order.",
                "properties": {
                    "quantity": "Units ordered.",
                    "price": "Price of one unit.",
                },
            },
            "Product": {
                "description": "An item the company sells; also called goods.",
                "properties": {"sku": "Stock keeping unit code."},
            },
            "Category": {"description": "A group of related products."},
            "Supplier": {
                "description": "A company that provides products to us; also called a vendor.",
                "properties": {"rating": "Quality rating of the supplier."},
            },
            "Warehouse": {
                "description": "A depot where products are stored.",
                "properties": {"capacity": "Units it can hold."},
            },
            "Invoice": {
                "description": "A bill issued to a customer for an order.",
                "properties": {
                    "amount": "Amount billed.",
                    "due": "Year payment is due.",
                },
            },
            "Payment": {
                "description": "Money received against an invoice.",
                "properties": {
                    "amount": "Amount received.",
                    "paid": "Year it was paid.",
                },
            },
            "Contract": {
                "description": "An agreement signed with a customer.",
                "properties": {
                    "value": "Total value of the agreement.",
                    "signed": "Year it was signed.",
                },
            },
            "Ticket": {
                "description": "A request for help raised by a customer; also called a support case.",
                "properties": {
                    "priority": "How urgent it is.",
                    "opened": "Year it was opened.",
                },
            },
            "Incident": {
                "description": "A problem or outage that caused one or more tickets.",
                "properties": {"severity": "From 1 (minor) to 5 (critical)."},
            },
            "Asset": {
                "description": "A piece of equipment that belongs to the company.",
                "properties": {"tag": "Inventory tag.", "cost": "Purchase cost."},
            },
            "Device": {
                "description": "A physical device that is registered as an asset.",
                "properties": {"model": "Kind of device, such as 'laptop'."},
            },
            "Repository": {
                "description": "A store of source code.",
                "properties": {"language": "Main programming language."},
            },
            "Commit": {
                "description": "A change recorded in a repository.",
                "properties": {"message": "The author's description of the change."},
            },
            "Release": {
                "description": "A published version of a repository.",
                "properties": {"version": "Version number."},
            },
            "Meeting": {"description": "A scheduled gathering of employees."},
            "Document": {
                "description": "A written document.",
                "properties": {"pages": "Number of pages."},
            },
        },
        "relationships": {
            "WORKS_IN": {
                "description": "An employee works in a department.",
                "properties": {"since": "Year the employee joined the department."},
            },
            "MEMBER_OF": {"description": "An employee belongs to a team."},
            "REPORTS_TO": {
                "description": "An employee reports to another employee, their line manager or boss."
            },
            "HAS_ROLE": {"description": "An employee holds a job position."},
            "HAS_SKILL": {
                "description": "An employee has a skill.",
                "properties": {"level": "Proficiency from 1 to 5."},
            },
            "HOLDS": {"description": "An employee has earned a certification."},
            "SITS_IN": {"description": "An employee works from an office."},
            "ASSIGNED_TO": {
                "description": "An employee is assigned to a task, or a ticket is assigned to the employee who handles it.",
                "properties": {"hours": "Hours the employee spent on the task."},
            },
            "MANAGES": {"description": "An employee leads a project."},
            "AUTHORED": {"description": "An employee wrote a commit."},
            "ATTENDED": {"description": "An employee took part in a meeting."},
            "OWNS": {"description": "An employee is responsible for an asset."},
            "PART_OF": {"description": "A department is part of a larger department."},
            "FUNDS": {
                "description": "A department pays for a project.",
                "properties": {"amount": "Money provided."},
            },
            "BELONGS_TO": {"description": "A team belongs to a department."},
            "LOCATED_IN": {
                "description": "An office is in a city, or a customer is based in a country."
            },
            "IN_COUNTRY": {"description": "A city is in a country."},
            "IN_REGION": {"description": "A country is in a region."},
            "HAS_TASK": {"description": "A project is made of tasks."},
            "HAS_MILESTONE": {"description": "A project has a milestone to reach."},
            "FOR_CUSTOMER": {"description": "A project is delivered for a customer."},
            "USES_REPOSITORY": {
                "description": "A project keeps its code in a repository."
            },
            "DEPENDS_ON": {
                "description": "A task cannot start before another task is done."
            },
            "HAS_COMMIT": {"description": "A repository contains a commit."},
            "HAS_RELEASE": {"description": "A repository has a published release."},
            "HAS_ACCOUNT": {"description": "A customer holds an account."},
            "HAS_CONTACT": {
                "description": "A person at a customer can be contacted.",
                "properties": {"primary": "Whether this is the main contact."},
            },
            "PLACED": {"description": "A customer placed an order."},
            "SIGNED": {"description": "A customer signed a contract."},
            "RAISED": {"description": "A customer opened a ticket."},
            "CONVERTED_TO": {"description": "A lead became a customer."},
            "FROM_CAMPAIGN": {"description": "A lead was brought in by a campaign."},
            "CONTAINS": {"description": "An order consists of order lines."},
            "OF_PRODUCT": {"description": "An order line is for a product."},
            "BILLED_BY": {"description": "An order is billed with an invoice."},
            "SETTLED_BY": {"description": "An invoice is paid by a payment."},
            "SHIPPED_FROM": {
                "description": "An order left a warehouse.",
                "properties": {"date": "Year it shipped."},
            },
            "IN_CATEGORY": {"description": "A product belongs to a category."},
            "SUPPLIED_BY": {
                "description": "A product is provided by a supplier.",
                "properties": {"cost": "What the supplier charges for one unit."},
            },
            "SUBCATEGORY_OF": {
                "description": "A category is inside a broader category."
            },
            "STOCKS": {
                "description": "A warehouse keeps a product in stock.",
                "properties": {"quantity": "Units in stock."},
            },
            "CAUSED_BY": {"description": "A ticket was caused by an incident."},
            "REGISTERED_AS": {"description": "A device is recorded as an asset."},
            "ABOUT": {"description": "A meeting was about a document."},
        },
    }
)
