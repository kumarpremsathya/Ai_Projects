import os
from dotenv import load_dotenv

load_dotenv()

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# response = llm.invoke("Hello")
from typing_extensions import TypedDict


class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    response: str


from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate

workflow = StateGraph(State)


def categorize_experience(state: State) -> State:
    print("\nCategorizing the experience level of candidate : ")
    prompt = ChatPromptTemplate.from_template(
        "Based on the following job application, categorize the candidate as 'Entry-level', 'Mid-level' or 'Senior-level'. "
        "Respond with ONLY one of: Entry-level, Mid-level, Senior-level. "
        "Application : {application}"
    )
    chain = prompt | llm
    experience_level = chain.invoke({"application": state["application"]}).content.strip()
    print(f"Experience Level : {experience_level}")
    return {"experience_level": experience_level}


def assess_skillset(state: State) -> State:
    print("\nAssessing the skillset of candidate : ")
    prompt = ChatPromptTemplate.from_template(
        "Based on the job application for a Python Developer, assess the candidate's skillset. "
        "Respond with ONLY one of: Match, No Match. "
        "Application : {application}"
    )
    chain = prompt | llm
    skill_match = chain.invoke({"application": state["application"]}).content.strip()
    print(f"Skill Match : {skill_match}")
    return {"skill_match": skill_match}


def schedule_hr_interview(state: State) -> State:
    print("\nScheduling the interview : ")
    return {"response": "Candidate has been shortlisted for an HR interview."}


def escalate_to_recruiter(state: State) -> State:
    print("Escalating to recruiter")
    return {"response": "Candidate has senior-level experience but doesn't match job skills."}


def reject_application(state: State) -> State:
    print("Sending rejecting email")
    return {"response": "Candidate doesn't meet JD and has been rejected."}


workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("assess_skillset", assess_skillset)
workflow.add_node("schedule_hr_interview", schedule_hr_interview)
workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
workflow.add_node("reject_application", reject_application)


def route_app(state: State) -> str:
    if state["skill_match"] == "Match":
        return "schedule_hr_interview"
    elif state["experience_level"] == "Senior-level":
        return "escalate_to_recruiter"
    else:
        return "reject_application"


workflow.add_edge(START, "categorize_experience")
workflow.add_edge("categorize_experience", "assess_skillset")
workflow.add_conditional_edges("assess_skillset", route_app)
workflow.add_edge("escalate_to_recruiter", END)
workflow.add_edge("reject_application", END)
workflow.add_edge("schedule_hr_interview", END)

app = workflow.compile()


def run_candidate_screening(application: str):
    results = app.invoke({"application": application})
    return {
        "experience_level": results["experience_level"],
        "skill_match": results["skill_match"],
        "response": results["response"],
    }


# application_text = "I have 10 years of experience in software engineering with expertise in JAVA"
# results = run_candidate_screening(application_text)
# print("\n\nComputed Results :")
# print(f"Application: {application_text}")
# print(f"Experience Level: {results['experience_level']}")
# print(f"Skill Match: {results['skill_match']}")
# print(f"Response: {results['response']}")



application_text = "I have 5 years of experience in software engineering with expertise in C++"
results = run_candidate_screening(application_text)
print("\n\nComputed Results :")
print(f"Application: {application_text}")
print(f"Experience Level: {results['experience_level']}")
print(f"Skill Match: {results['skill_match']}")
print(f"Response: {results['response']}")





# import os
# os.environ['OPENAI_API_KEY'] = 

# from langchain_openai.chat_models import ChatOpenAI

# llm = ChatOpenAI()

# from typing_extensions import TypedDict
# class State(TypedDict):
#   application: str
#   experience_level: str
#   skill_match : str
#   response: str


# from langgraph.graph import StateGraph, START, END

# workflow = StateGraph(State)

# from langchain_core.prompts import ChatPromptTemplate

# def categorize_experience(state: State) -> State:
#   print("\nCategorizing the experience level of candidate : ")
#   prompt = ChatPromptTemplate.from_template(
#       "Based on the following job application, categorize the candidate as 'Entry-level', 'Mid-level' or 'Senior-level'"
#       "Application : {application}"
#   )
#   chain = prompt | llm
#   experience_level = chain.invoke({"application": state["application"]}).content
#   print(f"Experience Level : {experience_level}")
#   return {"experience_level" : experience_level}

# def assess_skillset(state: State) -> State:
#   print("\nAssessing the skillset of candidate : ")
#   prompt = ChatPromptTemplate.from_template(
#       "Based on the job application for a Python Developer, assess the candidate's skillset"
#       "Respond with either 'Match' or 'No Match'"
#       "Application : {application}"
#   )
#   chain = prompt | llm
#   skill_match = chain.invoke({"application": state["application"]}).content
#   print(f"Skill Match : {skill_match}")
#   return {"skill_match" : skill_match}

# def schedule_hr_interview(state: State) -> State:
#   print("\nScheduling the interview : ")
#   return {"response" : "Candidate has been shortlisted for an HR interview."}

# def escalate_to_recruiter(state: State) -> State:
#   print("Escalating to recruiter")
#   return {"response" : "Candidate has senior-level experience but doesn't match job skills."}

# def reject_application(state: State) -> State:
#   print("Sending rejecting email")
#   return {"response" : "Candidate doesn't meet JD and has been rejected."}


# workflow.add_node("categorize_experience", categorize_experience)
# workflow.add_node("assess_skillset", assess_skillset)
# workflow.add_node("schedule_hr_interview", schedule_hr_interview)
# workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
# workflow.add_node("reject_application", reject_application)


# def route_app(state: State) -> str:
#   if(state["skill_match"] == "Match"):
#     return "schedule_hr_interview"
#   elif(state["experience_level"] == "Senior-level"):
#     return "escalate_to_recruiter"
#   else:
#     return "reject_application"
  
# workflow.add_edge("categorize_experience", "assess_skillset")
# workflow.add_conditional_edges("assess_skillset", route_app)

# workflow.add_edge(START, "categorize_experience")
# workflow.add_edge("assess_skillset", END)
# workflow.add_edge("escalate_to_recruiter", END)
# workflow.add_edge("reject_application", END)
# workflow.add_edge("schedule_hr_interview", END)

# app = workflow.compile()

# from IPython.display import Image, display

# display(
#     Image(
#         app.get_graph().draw_mermaid_png()
#     )
# )


# def run_candidate_screening(application: str):
#   results = app.invoke({"application" : application})
#   return {
#       "experience_level" : results["experience_level"],
#       "skill_match" : results["skill_match"],
#       "response" : results["response"]
#   }

# application_text = "I have 10 years of experience in software engineering with expertise in JAVA"
# results = run_candidate_screening(application_text)
# print("\n\nComputed Results :")
# print(f"Application: {application_text}")
# print(f"Experience Level: {results['experience_level']}")
# print(f"Skill Match: {results['skill_match']}")
# print(f"Response: {results['response']}")


# # application_text = "I have 1 year of experience in software engineering with expertise in JAVA"
# # results = run_candidate_screening(application_text)
# # print("\n\nComputed Results :")
# # print(f"Application: {application_text}")
# # print(f"Experience Level: {results['experience_level']}")
# # print(f"Skill Match: {results['skill_match']}")
# # print(f"Response: {results['response']}")


# # application_text = "I have experience in software engineering with expertise in Python"
# # results = run_candidate_screening(application_text)
# # print("\n\nComputed Results :")
# # print(f"Application: {application_text}")
# # print(f"Experience Level: {results['experience_level']}")
# # print(f"Skill Match: {results['skill_match']}")
# # print(f"Response: {results['response']}")

# # application_text = "I have 5 years of experience in software engineering with expertise in C++"
# # results = run_candidate_screening(application_text)
# # print("\n\nComputed Results :")
# # print(f"Application: {application_text}")
# # print(f"Experience Level: {results['experience_level']}")
# # print(f"Skill Match: {results['skill_match']}")
# # print(f"Response: {results['response']}")


