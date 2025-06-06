import os

from dotenv import load_dotenv
load_dotenv()

profile = {
    "name": "Unnati",
    "full_name": "Unnati Bamania",
    "user_profile_background": "Software Development Manager leading a team of 10 developers"
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage Unnati's tasks efficiently."
}

email = {
    "from": "John Smith <john.smith@company.com>",
    "to": "Unnati Bamania <unnati.bamania@company.com>",
    "subject": "Quick question about API documentation",
    "body": """

    Hi Unnati,

    I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

    Thanks!
    Unnati
    """,
}



from pydantic import  BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model


llm = init_chat_model(model="gpt-4o-mini")

class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )


llm_router = llm.with_structured_output(Router)

from prompts import triage_system_prompt, triage_user_prompt

print(triage_system_prompt)
print("-" * 50)
print(triage_user_prompt)

system_prompt = triage_system_prompt.format(
    full_name = profile["full_name"],
    name = profile["name"],
    user_profile_background = profile["user_profile_background"],
    triage_no = prompt_instructions["triage_rules"]["ignore"],
    triage_notify = prompt_instructions["triage_rules"]["notify"],
    triage_email = prompt_instructions["triage_rules"]["respond"],
    examples = None
)

user_prompt = triage_user_prompt.format(
    author = email["from"],
    to = email["to"],
    subject = email["subject"],
    email_thread = email["body"]
)

result = llm_router.invoke(
    [
        {"role": "system", "content" : system_prompt},
        {"role": "user", "content" : user_prompt},
    ]
)

from langchain_core.tools import tool

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""

    # Email content placeholder to be implemented here

    return f"Email sent to {to} with subject '{subject}'"


@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int,
    preferred_day: str
) -> str:
    """Schedule a Meeting"""

    # Calender check and meeting scheduleing placeholder code to be implment here

    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calender_availability(day: str) -> str:
    """Check calender availability for the given day."""

    # Code to check the calander availability to be implmented here

    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

from prompts import agent_system_prompt

print(agent_system_prompt)

def create_prompt(state):
    return [
        {
            "role": "system",
            "content": agent_system_prompt.format(
                instructions = prompt_instructions["agent_instructions"],
                **profile
            )
        }
    ] + state["messages"]
    
from langgraph.prebuilt import create_react_agent

tools= [write_email, schedule_meeting, check_calender_availability]

agent = create_react_agent(
    model="gpt-4o-mini",
    tools=tools,
    prompt=create_prompt,    
)

response = agent.invoke(
    {
        "messages" : [
            {
                "role": "user",
                "content": "What is my availability for Tuesday?"
            }
        ]
    }    
)

response["messages"][-1].pretty_print()

from langgraph.graph import add_messages

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]


from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from IPython.display import display, Image


def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:

    author = state["email_input"]["author"]
    to = state["email_input"]["to"],
    subject = state["email_input"]["subject"],
    email_thread = state["email_input"]["body"]

    system_prompt = triage_system_prompt.format(
        full_name = profile["full_name"],
        name = profile["name"],
        user_profile_background = profile["user_profile_background"],
        triage_no = prompt_instructions["triage_rules"]["ignore"],
        triage_notify = prompt_instructions["triage_rules"]["notify"],
        triage_email = prompt_instructions["triage_rules"]["respond"],
        examples = None
    )

    user_prompt = triage_user_prompt.format(
        author = author,
        to = to,
        subject = subject,
        email_thread = email_thread
    )

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    
    return Command(goto=goto, update=update)


builder = StateGraph(State)

builder.add_node("triage_router", triage_router)
builder.add_node("response_agent", agent)

builder.add_edge(START, "triage_router")

email_agent = builder.compile()


display(Image(email_agent.get_graph(xray=True).draw_mermaid_png()))

email_input = {

    "author": "Marketing Team <marketing@amazingdeals.com>",
    "to": "Sushant Dhumak <sushant.dhumak@company.com>",
    "subject": "🔥 EXCLUSIVE OFFER: Limited Time Discount on Developer Tools! 🔥",
    "body": """Dear Valued Developer,

    Don't miss out on this INCREDIBLE opportunity! 

    🚀 For a LIMITED TIME ONLY, get 80% OFF on our Premium Developer Suite! 

    ✨ FEATURES:
    - Revolutionary AI-powered code completion
    - Cloud-based development environment
    - 24/7 customer support
    - And much more!

    💰 Regular Price: $999/month
    🎉 YOUR SPECIAL PRICE: Just $199/month!

    🕒 Hurry! This offer expires in:
    24 HOURS ONLY!

    Click here to claim your discount: https://amazingdeals.com/special-offer

    Best regards,
    Marketing Team
    ---
    To unsubscribe, click here
    """,
}

response = email_agent.invoke(
    {"email_input": email_input}
)

email_input = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "Sushant Dhumak <sushant.dhumak@company.com>",
    "subject": "Quick question about API documentation",
    "body": """
    
    Hi Sushant,

    I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

    Specifically, I'm looking at:
    - /auth/refresh
    - /auth/validate

    Thanks!
    Alice
    """,
}

response = email_agent.invoke(
    {"email_input": email_input}
)

for msg in response["messages"]:
    msg.pretty_print()


print(result)