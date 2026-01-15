"""
AI Mentor Agent for First-Generation College Students
Provides guidance on internships, networking, college systems, and learning roadmaps
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime, timedelta
from dateutil import parser
import os
import sys
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Global chat history
chat_history = []

# =============================================================================
# CUSTOM TOOLS FOR AI MENTOR
# =============================================================================

@tool
def get_current_datetime() -> str:
    """Return current date & time in Indian Standard Time (IST)."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    except ImportError:
        now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    return now.strftime("%Y-%m-%d %H:%M:%S (IST)")

@tool
def search_internships(field: str, location: str = "India", experience_level: str = "beginner") -> str:
    """
    Search for internship opportunities in a specific field.
    
    Args:
        field: Field of interest (e.g., software engineering, marketing, data science)
        location: Preferred location (default: India)
        experience_level: Experience level (beginner, intermediate, advanced)
    
    Returns:
        Internship opportunities and application tips
    """
    try:
        query = f"internship opportunities for {experience_level} {field} students in {location} 2025 how to apply"
        
        tavily_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False
        )
        
        results = tavily_tool.invoke({"query": query})
        
        if results:
            formatted_results = f"🎯 Internship Opportunities in {field} ({location}):\n\n"
            if isinstance(results, list):
                for i, result in enumerate(results[:5], 1):
                    if isinstance(result, dict):
                        title = result.get('title', 'Opportunity')
                        content = result.get('content', '')
                        url = result.get('url', '')
                        formatted_results += f"{i}. {title}\n   {content[:250]}...\n   🔗 {url}\n\n"
            
            formatted_results += "\n💡 Application Tips:\n"
            formatted_results += "- Tailor your resume for each application\n"
            formatted_results += "- Highlight relevant projects and coursework\n"
            formatted_results += "- Apply early and follow up\n"
            formatted_results += "- Prepare for technical/behavioral interviews\n"
            
            return formatted_results
        else:
            return f"Consider exploring platforms like Internshala, LinkedIn, and company career pages for {field} internships in {location}."
            
    except Exception as e:
        return f"Error searching internships: {str(e)}. Try checking Internshala, LinkedIn, and Naukri.com for opportunities."

@tool
def search_scholarships(category: str = "general", field: str = "any") -> str:
    """
    Search for scholarship opportunities for first-generation students.
    
    Args:
        category: Scholarship category (merit, need-based, minority, women, general)
        field: Field of study (STEM, arts, business, etc.)
    
    Returns:
        Scholarship opportunities and eligibility criteria
    """
    try:
        query = f"{category} scholarships for first-generation college students {field} India 2025 eligibility how to apply"
        
        tavily_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True
        )
        
        results = tavily_tool.invoke({"query": query})
        
        if results:
            formatted_results = f"💰 Scholarship Opportunities ({category} - {field}):\n\n"
            if isinstance(results, list):
                for i, result in enumerate(results[:5], 1):
                    if isinstance(result, dict):
                        title = result.get('title', 'Scholarship')
                        content = result.get('content', '')
                        url = result.get('url', '')
                        formatted_results += f"{i}. {title}\n   {content[:250]}...\n   🔗 {url}\n\n"
            return formatted_results
        else:
            return "Explore government scholarships (NSP, AICTE), private scholarships, and university-specific aid programs."
            
    except Exception as e:
        return f"Error searching scholarships: {str(e)}. Check National Scholarship Portal and your institution's financial aid office."

@tool
def get_learning_roadmap(topic: str, current_level: str = "beginner") -> str:
    """
    Generate a comprehensive learning roadmap for any topic with resources.
    
    Args:
        topic: The subject/skill to learn (e.g., Python, Data Science, Web Development)
        current_level: Current skill level (beginner, intermediate, advanced)
    
    Returns:
        Detailed learning path with stages, resources, and timeline
    """
    try:
        # Search for learning resources
        query = f"complete {topic} learning roadmap for {current_level} free courses tutorials resources 2025"
        
        tavily_tool = TavilySearchResults(
            max_results=6,
            search_depth="advanced",
            include_answer=True
        )
        
        results = tavily_tool.invoke({"query": query})
        
        roadmap = f"🗺️ Learning Roadmap: {topic}\n"
        roadmap += f"📊 Starting Level: {current_level.title()}\n\n"
        
        # Define stages based on level
        if current_level.lower() == "beginner":
            stages = [
                ("Stage 1: Foundations", "1-2 months"),
                ("Stage 2: Core Concepts", "2-3 months"),
                ("Stage 3: Practical Application", "2-3 months"),
                ("Stage 4: Advanced Topics", "3-4 months")
            ]
        elif current_level.lower() == "intermediate":
            stages = [
                ("Stage 1: Review & Strengthen", "2-3 weeks"),
                ("Stage 2: Advanced Concepts", "2-3 months"),
                ("Stage 3: Specialization", "3-4 months"),
                ("Stage 4: Expert Projects", "2-3 months")
            ]
        else:
            stages = [
                ("Stage 1: Expert-Level Topics", "2-3 months"),
                ("Stage 2: Research & Innovation", "3-4 months"),
                ("Stage 3: Contribution & Teaching", "Ongoing")
            ]
        
        for stage, duration in stages:
            roadmap += f"\n{'='*60}\n"
            roadmap += f"{stage} ({duration})\n"
            roadmap += f"{'='*60}\n"
        
        roadmap += "\n\n📚 Recommended Learning Resources:\n\n"
        
        if results and isinstance(results, list):
            for i, result in enumerate(results[:6], 1):
                if isinstance(result, dict):
                    title = result.get('title', 'Resource')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    roadmap += f"{i}. {title}\n"
                    roadmap += f"   📝 {content[:200]}...\n"
                    roadmap += f"   🔗 {url}\n\n"
        
        roadmap += "\n💡 Learning Tips:\n"
        roadmap += "✓ Set aside dedicated study time daily (1-2 hours minimum)\n"
        roadmap += "✓ Build projects to apply what you learn\n"
        roadmap += "✓ Join online communities (Reddit, Discord, Stack Overflow)\n"
        roadmap += "✓ Track your progress and celebrate milestones\n"
        roadmap += "✓ Don't hesitate to revisit fundamentals\n"
        roadmap += "✓ Teach others - it reinforces your learning\n"
        
        return roadmap
        
    except Exception as e:
        return f"Error generating roadmap: {str(e)}. I recommend starting with free platforms like Coursera, edX, and YouTube."

@tool
def draft_professional_email(purpose: str, recipient: str, context: str = "") -> str:
    """
    Draft a professional email template.
    
    Args:
        purpose: Purpose of email (internship inquiry, professor meeting, networking, etc.)
        recipient: Who you're emailing (professor, recruiter, professional, etc.)
        context: Additional context or specific points to include
    
    Returns:
        Professional email template with guidance
    """
    templates = {
        "internship": {
            "subject": "Application for [Position Name] - [Your Name]",
            "body": """Dear [Recipient Name],

I hope this email finds you well. My name is [Your Name], and I am a [Year] year student pursuing [Your Degree] at [Your College]. I am writing to express my strong interest in the [Position Name] position at [Company Name].

{context_section}

I am particularly drawn to [Company Name] because [specific reason related to company/role]. Through my coursework and projects, I have developed skills in [relevant skills], which I believe align well with the requirements of this position.

I have attached my resume for your review. I would be grateful for the opportunity to discuss how I can contribute to your team. I am available for an interview at your convenience.

Thank you for considering my application. I look forward to hearing from you.

Best regards,
[Your Name]
[Your Phone Number]
[Your Email]
[LinkedIn Profile]"""
        },
        "professor": {
            "subject": "Request for Meeting - [Your Name] from [Course Name]",
            "body": """Dear Professor [Last Name],

I hope you are doing well. My name is [Your Name], and I am a student in your [Course Name] class ([Section/Time]).

{context_section}

I was wondering if you might have time for a brief meeting to discuss [specific topic]. I am available [mention your availability] and would be happy to meet at your convenience, whether in person during office hours or via video call.

Thank you very much for your time and consideration.

Respectfully,
[Your Name]
[Student ID]
[Your Email]"""
        },
        "networking": {
            "subject": "Seeking Advice from [Industry] Professional - [Your Name]",
            "body": """Dear [Recipient Name],

I hope this message finds you well. My name is [Your Name], and I am a [Year] year student at [Your College] studying [Your Major]. I came across your profile on [LinkedIn/other platform] and was impressed by your work in [specific area].

{context_section}

I am eager to learn more about [specific field/topic] and would greatly appreciate the opportunity to hear about your experiences and any advice you might have for someone starting their career in this field.

Would you be available for a brief 15-20 minute informational interview, either via phone or video call? I am flexible with timing and happy to work around your schedule.

Thank you for considering my request. I understand you have a busy schedule and would be grateful for any time you could spare.

Best regards,
[Your Name]
[Your Email]
[LinkedIn Profile]"""
        }
    }
    
    template_key = "internship"
    for key in templates:
        if key in purpose.lower():
            template_key = key
            break
    
    template = templates[template_key]
    context_section = f"\n{context}\n" if context else "\n[Mention relevant experience, skills, or why you're reaching out]\n"
    
    email_draft = f"📧 Professional Email Template\n\n"
    email_draft += f"Subject: {template['subject']}\n\n"
    email_draft += template['body'].replace("{context_section}", context_section)
    email_draft += "\n\n💡 Email Tips:\n"
    email_draft += "✓ Keep it concise (under 200 words)\n"
    email_draft += "✓ Proofread for grammar and spelling\n"
    email_draft += "✓ Use a professional email address\n"
    email_draft += "✓ Follow up after 5-7 days if no response\n"
    email_draft += "✓ Be respectful of their time\n"
    
    return email_draft

@tool
def optimize_linkedin_profile(section: str, field: str = "general") -> str:
    """
    Get tips to optimize LinkedIn profile sections.
    
    Args:
        section: Profile section (headline, summary, experience, skills, etc.)
        field: Your field of study/interest
    
    Returns:
        Optimization tips and examples
    """
    tips = {
        "headline": f"""🎯 LinkedIn Headline Optimization for {field}:

Your headline should be more than just your title. Use this formula:
[Your Role/Status] | [Key Skills] | [What You're Passionate About]

Examples:
• Computer Science Student | Python & Machine Learning | Aspiring Data Scientist
• Business Major | Marketing & Analytics | Passionate About Digital Strategy
• Engineering Student | IoT & Robotics Enthusiast | Building Smart Solutions

Tips:
✓ Include relevant keywords for your field
✓ Show what makes you unique
✓ Keep it under 120 characters
✓ Update it as you grow
""",
        "summary": f"""📝 LinkedIn Summary Guide for {field}:

Your summary should tell your story in 3-5 paragraphs:

Paragraph 1: Who you are and your current focus
"I'm a [year] year [major] student at [college], passionate about [field]..."

Paragraph 2: Your key experiences and skills
"Through my coursework and projects, I've developed expertise in..."

Paragraph 3: What you're looking for
"I'm currently seeking [internships/opportunities] in [specific area]..."

Paragraph 4: Your values/interests
"I'm particularly interested in [specific topic] and enjoy [relevant activities]..."

Closing: Call to action
"Feel free to connect if you'd like to discuss [topic] or explore collaboration opportunities!"

Tips:
✓ Write in first person
✓ Use short paragraphs
✓ Include keywords naturally
✓ Show personality
✓ Update regularly
""",
        "experience": f"""💼 Experience Section Tips for {field}:

For each position/project, use this structure:

Title: [Be specific and professional]
Example: "Software Development Intern" not just "Intern"

Description: Use the STAR method
• Situation: Brief context
• Task: What you needed to do
• Action: What you actually did (use action verbs)
• Result: Quantifiable outcomes

Action Verbs for {field}:
Developed, Created, Implemented, Analyzed, Designed, Managed, Led, Optimized, Improved, Collaborated

Example:
"Developed a web application using React and Node.js that improved user engagement by 40%. Collaborated with a team of 4 to implement features and conducted code reviews."

Tips:
✓ Quantify achievements with numbers
✓ Start each point with an action verb
✓ Include relevant technologies/tools
✓ List most recent first
""",
        "skills": f"""🛠️ Skills Section Optimization for {field}:

Organize skills by category:

Technical Skills:
• List programming languages, tools, frameworks
• Get endorsed by classmates and colleagues

Soft Skills:
• Communication, Leadership, Problem-solving, Teamwork

Domain Skills:
• Field-specific competencies

Priority Order:
1. Most relevant skills for your target role (top 3)
2. Skills you're currently developing
3. Foundational skills

Tips:
✓ Add 10-50 skills (sweet spot: 25-30)
✓ Take skill assessments
✓ Reorder based on job applications
✓ Request endorsements from peers
✓ Include both technical and soft skills
"""
    }
    
    section_lower = section.lower()
    for key in tips:
        if key in section_lower:
            return tips[key]
    
    return f"LinkedIn profile tip: Focus on making your profile complete, professional, and keyword-rich for {field}. Update all sections regularly!"

@tool
def explain_college_process(process: str, context: str = "India") -> str:
    """
    Explain college administrative processes and systems.
    
    Args:
        process: Process to explain (registration, grades, financial aid, etc.)
        context: Educational context (default: India)
    
    Returns:
        Detailed explanation of the process
    """
    try:
        query = f"college {process} process guide for students {context} step by step how to"
        
        tavily_tool = TavilySearchResults(
            max_results=3,
            search_depth="advanced",
            include_answer=True
        )
        
        results = tavily_tool.invoke({"query": query})
        
        explanation = f"📋 Understanding: {process.title()}\n\n"
        
        if results and isinstance(results, list):
            for result in results[:3]:
                if isinstance(result, dict):
                    content = result.get('content', '')
                    url = result.get('url', '')
                    if content:
                        explanation += f"{content[:300]}...\n\n"
                        explanation += f"Source: {url}\n\n"
        
        explanation += "💡 General Tips:\n"
        explanation += "✓ Read your college handbook/website thoroughly\n"
        explanation += "✓ Contact your academic advisor for personalized guidance\n"
        explanation += "✓ Mark important deadlines in your calendar\n"
        explanation += "✓ Keep copies of all important documents\n"
        explanation += "✓ Don't hesitate to ask for help from college staff\n"
        
        return explanation
        
    except Exception as e:
        return f"For information about {process}, please check your college's official website or contact the administrative office. They can provide specific guidance for your situation."

@tool
def get_deadline_reminders(event_type: str = "general") -> str:
    """
    Get common academic deadlines and reminder tips.
    
    Args:
        event_type: Type of deadline (registration, financial_aid, career, applications)
    
    Returns:
        Important deadlines and how to track them
    """
    current_date = datetime.now()
    month = current_date.month
    
    deadlines_info = f"📅 Important Deadlines & Reminders ({event_type.title()})\n\n"
    
    deadlines_by_type = {
        "registration": [
            "Course Registration: Usually 2-4 weeks before semester starts",
            "Add/Drop Period: First 1-2 weeks of semester",
            "Late Registration Fee Deadline: Check your college calendar",
            "Major Declaration: Typically end of sophomore year"
        ],
        "financial_aid": [
            "Scholarship Applications: Varies by scholarship (often fall semester)",
            "FAFSA/Government Aid: Check national/state deadlines",
            "Tuition Payment: Before semester starts",
            "Work-Study Applications: Early in semester"
        ],
        "career": [
            "Career Fair Registration: 2-3 weeks before event",
            "Summer Internship Applications: October - February",
            "Resume Review Sessions: Ongoing, check career center",
            "On-Campus Recruitment: Varies by company"
        ],
        "applications": [
            "Graduate School Applications: September - December",
            "Study Abroad Programs: 6-12 months in advance",
            "Research Opportunities: Rolling basis",
            "Leadership Positions: Varies by organization"
        ]
    }
    
    relevant_deadlines = deadlines_by_type.get(event_type, deadlines_by_type["registration"])
    
    for deadline in relevant_deadlines:
        deadlines_info += f"• {deadline}\n"
    
    deadlines_info += "\n\n🔔 Deadline Management Tips:\n"
    deadlines_info += "1. Use a digital calendar (Google Calendar) with notifications\n"
    deadlines_info += "2. Set multiple reminders (1 month, 1 week, 1 day before)\n"
    deadlines_info += "3. Create a semester timeline at the start\n"
    deadlines_info += "4. Subscribe to college newsletter/announcements\n"
    deadlines_info += "5. Join student groups for peer reminders\n"
    deadlines_info += "6. Check your college portal weekly\n"
    
    return deadlines_info

# =============================================================================
# CREATE AGENT
# =============================================================================

def create_agent():
    """Initialize and return the AI mentor agent executor."""
    
    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=3000,
        timeout=60,
        max_retries=2
    )
    
    # Initialize Tavily search
    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False
    )
    
    # Define all tools
    tools = [
        get_current_datetime,
        search_internships,
        search_scholarships,
        get_learning_roadmap,
        draft_professional_email,
        optimize_linkedin_profile,
        explain_college_process,
        get_deadline_reminders,
        tavily_tool
    ]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI Mentor specifically designed for first-generation college students. You provide emotional support, practical guidance, and actionable advice on navigating college life, career development, and personal growth.

Your core capabilities:
1. **Life Events Agent**: Track deadlines, explain college processes, help with academic planning
2. **Opportunity Discovery Agent**: Find internships, scholarships, networking events
3. **Communication Coach Agent**: Draft emails, optimize LinkedIn, prepare for interviews
4. **Learning Roadmap Agent**: Create personalized learning paths for any subject/skill

Available Tools - Use them proactively:
• search_internships: Find internship opportunities
• search_scholarships: Discover scholarship programs
• get_learning_roadmap: Generate comprehensive learning paths with resources
• draft_professional_email: Create professional email templates
• optimize_linkedin_profile: Provide LinkedIn optimization tips
• explain_college_process: Explain college administrative processes
• get_deadline_reminders: Provide deadline tracking tips
• tavily_search_results_json: Search the web for current information

Your approach:
- Be warm, encouraging, and empathetic
- Acknowledge the unique challenges of first-generation students
- Provide specific, actionable advice
- Break down complex processes into simple steps
- Consider the Indian educational context when relevant
- Always offer resources and next steps
- Celebrate small wins and progress
- Never make students feel inadequate

When students ask about learning a topic:
1. Use get_learning_roadmap tool to create a structured path
2. Include beginner to advanced stages
3. Provide free and paid resource links
4. Suggest practical projects
5. Give realistic timelines
6. Add motivational tips

Be proactive:
- If someone mentions wanting an internship, offer to search
- If they need to email someone, offer to draft it
- If they're confused about a process, explain it clearly
- If they want to learn something, create a complete roadmap

Remember: You're not just providing information - you're being a supportive mentor who truly cares about their success."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        max_execution_time=180
    )
    
    return agent_executor

# =============================================================================
# CHAT FUNCTION
# =============================================================================

def chat(user_input: str, agent_executor):
    """
    Process user input and maintain chat history.
    
    Args:
        user_input: The user's message
        agent_executor: The agent executor instance
    
    Returns:
        The agent's response
    """
    global chat_history
    
    try:
        if chat_history is None:
            chat_history = []
        
        # Format chat history
        formatted_history = []
        for msg in chat_history:
            if isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role == "human":
                    formatted_history.append(HumanMessage(content=content))
                elif role == "assistant" and content:
                    if isinstance(content, str):
                        formatted_history.append(AIMessage(content=content))
        
        if agent_executor is None:
            agent_executor = create_agent()
        
        # Prepare input
        input_data = {
            "input": user_input,
            "chat_history": formatted_history or []
        }
        
        # Run agent
        try:
            response = agent_executor.invoke(input_data)
            
            if response is None:
                output = "I'm here to help! Could you please rephrase your question or tell me more about what you need assistance with?"
            elif isinstance(response, dict):
                output = response.get('output', '')
                if not output:
                    output = "I'm having a bit of trouble understanding. Could you provide more details about what you need help with?"
            elif hasattr(response, 'output') and response.output is not None:
                output = str(response.output)
            else:
                output = str(response) if response is not None else "I'm here to help! What would you like to know?"
            
            if not output or not isinstance(output, str):
                output = "I want to help you! Could you tell me more about what you're looking for?"
                
        except Exception as e:
            output = f"I encountered a technical issue, but I'm still here to help! Could you try rephrasing your question? Error: {str(e)}"
        
        # Update chat history
        if output and output != 'No response generated':
            chat_history.append(("human", user_input))
            chat_history.append(("assistant", output))
        
        # Keep last 30 messages for context
        if len(chat_history) > 30:
            chat_history = chat_history[-30:]
        
        return output if output else "I'm here as your mentor! How can I support you today?"
        
    except Exception as e:
        error_msg = f"I encountered an error, but don't worry - I'm still here to help: {str(e)}"
        print(error_msg)
        return "I'm having a technical issue, but please try asking your question again. I'm here to support you!"

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🎓 AI Mentor for First-Generation College Students")
    print("=" * 70)
    print("\n📋 Loading API keys from .env file...")
    print("\nRequired API Keys:")
    print("- GROQ_API_KEY (for LLM)")
    print("- TAVILY_API_KEY (for web search)")
    print("=" * 70)
    
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️  GROQ_API_KEY not found in .env file!")
        print("\nPlease create a .env file with:")
        print("GROQ_API_KEY=gsk-your-groq-key-here")
        print("TAVILY_API_KEY=tvly-your-tavily-key-here")
        sys.exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("\n⚠️  TAVILY_API_KEY not found in .env file!")
        sys.exit(1)
    
    print("✅ API keys loaded successfully!")
    
    print("\n🤖 Initializing AI Mentor agent...")
    try:
        agent_executor = create_agent()
        print("✅ AI Mentor ready!\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {str(e)}")
        sys.exit(1)
    
    print("Chat with your AI Mentor (type 'quit' to exit):\n")
    print("Examples:")
    print("- 'I want to learn Python programming from scratch'")
    print("- 'Help me find internships in data science'")
    print("- 'Draft an email to a professor for research opportunity'")
    print("- 'How do I register for courses?'\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nTake care! Remember, I'm always here when you need guidance. 🎓")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nTake care! Remember, I'm always here when you need guidance. 🎓")
            break
        
        if user_input.lower() == 'clear':
            chat_history.clear()
            print("\n✅ Chat history cleared!\n")
            continue
        
        try:
            response = chat(user_input, agent_executor)
            print(f"\n🎓 AI Mentor: {response}\n")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")