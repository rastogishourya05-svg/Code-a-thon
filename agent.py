"""
AI Mentor for First-Generation College Students
Multi-agent system providing guidance on academics, internships, networking, and life skills
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv
import json

load_dotenv()

# Global chat history
chat_history = []

# =============================================================================
# SPECIALIZED AGENTS TOOLS
# =============================================================================

@tool
def get_current_datetime() -> str:
    """
    Use this tool when the user asks what time or date it is.
    Returns current date & time in Indian Standard Time (IST).
    """
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    except ImportError:
        now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    return now.strftime("%Y-%m-%d %H:%M:%S (IST)")

# =============================================================================
# LIFE EVENTS AGENT TOOLS
# =============================================================================

@tool
def get_college_deadlines(event_type: str) -> str:
    """
    Use this tool when the user asks about college deadlines, important dates,
    academic calendar, or timelines for: admission, scholarship, exam, internship,
    or assignment submission.

    Args:
        event_type: One of - admission, scholarship, exam, internship, assignment
    """
    deadlines = {
        "admission": """
📅 ADMISSION DEADLINES:
• College Applications: November - January
• Document Verification: Within 7 days of admission
• Fee Payment: Within 15 days of admission
• Hostel Application: June - July for new session
• Course Registration: First week of semester

⚠️ Important Tips:
- Keep all documents ready in advance
- Set reminders 1 week before deadlines
- Follow up with admission office regularly
""",
        "scholarship": """
💰 SCHOLARSHIP DEADLINES:
• National Scholarship Portal: October - November
• Merit-based Scholarships: August - September
• Need-based Scholarships: Throughout the year
• State Scholarships: Check state portal monthly
• College-specific Scholarships: Ask student office

📋 Required Documents:
- Income certificate (< 6 months old)
- Caste certificate (if applicable)
- Previous year marksheets
- Bank account details
- Aadhaar card
""",
        "exam": """
📚 EXAM PREPARATION TIMELINE:
• Mid-semester Exams: 6-8 weeks into semester
• End-semester Exams: Last month of semester
• Internal Assessments: Ongoing throughout
• Project Submissions: Check syllabus for dates

⏰ Preparation Strategy:
- Start studying 3 weeks before exams
- Complete syllabus 1 week before
- Revise and practice 3-4 days before
- Don't skip sleep before exams
""",
        "internship": """
💼 INTERNSHIP TIMELINE:
• Summer Internships: Applications in Jan-March
• Winter Internships: Applications in Aug-Sept
• Company Drives: Throughout the year
• Application Deadlines: Usually 2-4 weeks notice

🎯 Action Items:
- Update resume by December/July
- Build LinkedIn profile early
- Start applying 3 months before summer/winter
- Practice common interview questions
""",
        "assignment": """
📝 ASSIGNMENT MANAGEMENT:
• Weekly Assignments: Submit within 7 days
• Monthly Projects: Plan 2 weeks in advance
• Semester Projects: Start 1 month early
• Lab Reports: Submit within 3-5 days

✅ Best Practices:
- Note down all deadlines immediately
- Start working at least 3 days early
- Keep backups of all work
- Ask doubts before last day
"""
    }

    event_lower = event_type.lower()
    for key in deadlines:
        if key in event_lower:
            return deadlines[key]

    return """
📅 GENERAL ACADEMIC CALENDAR:
• Semester Start: July/August & January
• Mid-term Break: October & March
• End-semester Exams: November-December & April-May
• Summer Break: May-July
• Winter Break: December-January

💡 Tips:
- Check college website for official calendar
- Join student WhatsApp groups for updates
- Meet academic advisor every semester
- Mark all important dates in your phone
"""

@tool
def explain_college_process(process_name: str) -> str:
    """
    Use this tool when the user asks HOW to do something college-related:
    course registration, hostel allocation, library membership, attendance tracking,
    understanding grades or CGPA, fee payment, ID card, or certificates.

    Args:
        process_name: The college process to explain. Examples: registration,
                      hostel, library, attendance, grades, fees, id card, certificate
    """
    processes = {
        "registration": """
📋 COURSE REGISTRATION PROCESS:

Step 1: Check Eligible Courses
- Review your semester curriculum
- Check prerequisites completed
- Note course codes and credits

Step 2: Online Registration
- Login to college portal with credentials
- Go to "Course Registration" section
- Select courses (maintain min/max credits)
- Add electives if allowed

Step 3: Confirm & Pay Fees
- Review your selected courses
- Download course registration form
- Pay semester fees (if required)
- Take confirmation receipt

Step 4: Attend Classes
- Check timetable on portal
- Note classroom locations
- Attend first class within 3 days
- Get syllabus from professor

⚠️ Common Mistakes to Avoid:
- Missing registration deadline
- Not checking prerequisites
- Overloading credits
- Forgetting to pay fees on time
""",
        "hostel": """
🏠 HOSTEL ALLOCATION PROCESS:

Step 1: Application
- Fill hostel application form
- Submit required documents
- Pay hostel fees advance
- Choose hostel preference (if option available)

Step 2: Allocation
- Wait for allocation list (usually 1-2 weeks)
- Check college notice board/website
- Note your room number and block

Step 3: Check-in
- Visit hostel office with documents
- Collect room keys and ID card
- Sign hostel rules acknowledgment
- Complete room inspection

Step 4: Settling In
- Meet your roommates
- Buy necessary items (bucket, lock, etc.)
- Note mess timings
- Get important contact numbers

📱 Important Contacts to Save:
- Hostel warden number
- Mess in-charge
- Security office
- Maintenance department
- Fellow hostel mates
""",
        "library": """
📚 LIBRARY SYSTEM GUIDE:

Step 1: Get Library Card
- Visit library with ID card
- Fill membership form
- Get library card issued
- Set up online account

Step 2: Finding Books
- Use online catalog (OPAC)
- Search by title/author/subject
- Note down call number
- Locate section in library

Step 3: Borrowing Books
- Take books to circulation desk
- Show library card
- Check due date (usually 2 weeks)
- Renew online if needed

Step 4: Digital Resources
- Access e-books and journals
- Use college WiFi or VPN
- Download research papers
- Use reference management tools

💡 Pro Tips:
- Reserve popular books online
- Set due date reminders
- Use study rooms for group work
- Ask librarian for research help
""",
        "attendance": """
✅ ATTENDANCE MANAGEMENT:

Understanding Rules:
- Minimum Required: Usually 75%
- Medical Leave: Doctor certificate needed
- Official Duty: Get prior approval letter
- Condonation: Available in some colleges (fees apply)

How to Track:
- Check college portal weekly
- Maintain personal attendance sheet
- Note dates you were absent
- Calculate percentage regularly

Formula: (Classes Attended / Total Classes) × 100

If Attendance is Low:
- Talk to professor immediately
- Submit valid reason documents
- Apply for condonation (if eligible)
- Attend extra lectures if offered
- Plan to attend all remaining classes

📊 Attendance Calculation Example:
- Total Classes: 40
- Attended: 32
- Percentage: 32/40 × 100 = 80% ✅

⚠️ Consequences of Low Attendance:
- May not be allowed to sit in exams
- Affects internal marks
- Can delay graduation
""",
        "grades": """
📊 GRADING SYSTEM EXPLAINED:

Grade Scale (Common System):
- O (Outstanding): 90-100% (10 points)
- A+ (Excellent): 80-89% (9 points)
- A (Very Good): 70-79% (8 points)
- B+ (Good): 60-69% (7 points)
- B (Above Average): 55-59% (6 points)
- C (Average): 50-54% (5 points)
- P (Pass): 45-49% (4 points)
- F (Fail): Below 45% (0 points)

How to Calculate CGPA:
1. Note credit points for each subject
2. Multiply grade points × credit points
3. Add all products
4. Divide by total credit points

Example:
Subject A: 8 grade × 4 credits = 32
Subject B: 9 grade × 3 credits = 27
Subject C: 7 grade × 3 credits = 21
Total: 80 / 10 credits = 8.0 CGPA

Understanding Components:
- Internal Assessment: 30-40%
- End Semester Exam: 60-70%
- Attendance: 5% (some colleges)
- Assignments/Quizzes: Included in internal

💡 How to Improve Grades:
- Focus on internal assessments
- Submit all assignments on time
- Participate in class discussions
- Clear doubts regularly
- Practice previous year papers
"""
    }

    process_lower = process_name.lower()
    for key in processes:
        if key in process_lower:
            return processes[key]

    return """
🎓 COMMON COLLEGE PROCESSES:

I can help explain:
• Course Registration - How to register for subjects
• Hostel Allocation - Getting hostel accommodation
• Library System - Using library resources
• Attendance Rules - Managing attendance
• Grades & CGPA - Understanding grading system
• Fee Payment - Paying college fees
• Exam Forms - Filling examination forms
• ID Card - Getting student ID
• Certificates - Requesting official documents

Please specify which process you'd like to learn about!
"""

# =============================================================================
# OPPORTUNITY DISCOVERY AGENT TOOLS
# =============================================================================

@tool
def search_internships(field: str, year: str, location: str = "India") -> str:
    """
    Use this tool when the user asks about internships, work experience,
    summer/winter placements, or job opportunities as a student.

    Args:
        field: Field of study e.g. computer science, engineering, business, design
        year: Year of study e.g. 1st year, 2nd year, 3rd year, 4th year
        location: Preferred location (default: India)
    """
    try:
        query = f"internship opportunities for {year} {field} students in {location} 2025"

        tavily_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False
        )

        results = tavily_tool.invoke({"query": query})

        base_info = f"""
💼 INTERNSHIP OPPORTUNITIES FOR {field.upper()} - {year}

🌐 Top Platforms to Find Internships:
1. Internshala (internshala.com) - Most popular in India
2. LinkedIn Jobs - Professional networking
3. AngelList - Startups and tech companies
4. LetsIntern - Various fields
5. Unstop (formerly Dare2Compete) - Competitions + Internships

📱 Company Career Pages:
- Google Careers
- Microsoft Student Programs
- Amazon Jobs
- Flipkart Careers
- Zomato Careers
- Check your dream companies' websites

"""

        if results and isinstance(results, list):
            base_info += "\n🔍 Latest Opportunities Found:\n\n"
            for i, result in enumerate(results[:3], 1):
                if isinstance(result, dict):
                    title = result.get('title', 'Opportunity')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    base_info += f"{i}. {title}\n   {content[:150]}...\n   Apply: {url}\n\n"

        base_info += """
✅ Application Tips:
• Start applying 3 months before summer/winter break
• Customize resume for each application
• Write personalized cover letters
• Follow up after 1 week if no response
• Prepare for common interview questions

📝 Required Documents:
- Updated resume (1-page)
- Cover letter template
- Recommendation letter (if available)
- Portfolio/GitHub (for technical roles)
- Previous internship certificates
"""

        return base_info

    except Exception as e:
        return f"""
💼 INTERNSHIP SEARCH - {field.upper()}

🌐 Top Platforms (Always Work):
1. Internshala.com - #1 platform in India
2. LinkedIn Jobs
3. Unstop.com
4. Company career pages directly

📊 Timeline for {year}:
• Start Date: 3 months before summer/winter
• Applications: Apply to 20-30 positions
• Follow-ups: After 1 week
• Interviews: Prepare 2 weeks in advance

Error accessing live data: {str(e)}
"""

@tool
def find_scholarships(category: str, state: str = "India") -> str:
    """
    Use this tool when the user asks about scholarships, financial aid,
    stipends, fee waivers, or money to support their education.

    Args:
        category: Student category e.g. general, sc, st, obc, minority, merit, need-based
        state: State name for state-specific scholarships (default: India)
    """
    try:
        query = f"{category} scholarships for college students {state} 2025 how to apply"

        tavily_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True
        )

        results = tavily_tool.invoke({"query": query})

        base_info = f"""
💰 SCHOLARSHIP OPPORTUNITIES - {category.upper()}

🎯 Major Scholarship Portals:
1. National Scholarship Portal (scholarships.gov.in)
   - All central government scholarships
   - Apply: October-November usually

2. State Scholarship Portals
   - Check your state education department website
   - Deadlines vary by state

3. College-Specific Scholarships
   - Ask your college scholarship cell
   - Usually merit or need-based

4. Private Scholarships:
   - Buddy4Study
   - Indian Oil Academic Scholarships
   - HDFC Educational Crisis Scholarship
   - Sitaram Jindal Foundation
   - K.C. Mahindra Scholarships

"""

        if results and isinstance(results, list):
            base_info += "\n🔍 Current Scholarship Programs:\n\n"
            for i, result in enumerate(results[:3], 1):
                if isinstance(result, dict):
                    title = result.get('title', 'Scholarship')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    base_info += f"{i}. {title}\n   {content[:150]}...\n   Link: {url}\n\n"

        base_info += """
📋 Common Required Documents:
✓ Income certificate (< 6 months old)
✓ Caste certificate (if applicable)
✓ Previous year marksheets (class 10, 12, college)
✓ Aadhaar card
✓ Bank account passbook (student's name)
✓ College ID card
✓ Fee receipt

⏰ General Timeline:
• Applications Open: August-November
• Document Submission: Within 15 days
• Verification: 1-2 months
• Disbursement: After verification

💡 Pro Tips:
• Apply early - don't wait for deadline
• Keep scanned copies ready
• Check application status weekly
• Keep mobile number/email updated
• Apply to multiple scholarships
"""

        return base_info

    except Exception as e:
        return f"""
💰 SCHOLARSHIP INFORMATION

🎯 Main Portals:
1. scholarships.gov.in - National portal
2. Your state scholarship portal
3. College scholarship cell
4. Buddy4Study.com

📋 Documents Needed:
- Income certificate
- Marksheets
- Aadhaar
- Bank details
- Caste certificate (if applicable)

Error accessing live data: {str(e)}
"""

@tool
def networking_opportunities(interest: str) -> str:
    """
    Use this tool when the user asks about networking, building professional
    connections, finding communities, attending events, or meeting people in their field.

    Args:
        interest: Area of interest e.g. tech, business, design, research, finance
    """
    opportunities = f"""
🤝 NETWORKING OPPORTUNITIES - {interest.upper()}

📱 Online Communities to Join:

LinkedIn Groups:
• {interest} Professionals India
• College to Corporate
• First-Generation Professionals
• Young Professionals Network

WhatsApp/Telegram Communities:
• College alumni groups
• {interest} enthusiasts groups
• Industry-specific channels
• Startup networking groups

Discord Servers:
• Programming communities (if tech)
• Design communities
• Startup ecosystems
• College project groups

🎪 Events to Attend:

1. College Events:
   • Guest lectures by alumni
   • Industry visits
   • Technical fests
   • Cultural events with industry presence

2. External Events:
   • Meetup.com - Local professional meetups
   • TechSparks, Surge (for tech)
   • Industry conferences (often have student passes)
   • Hackathons and competitions

3. Virtual Events:
   • Webinars by industry leaders
   • Twitter Spaces
   • LinkedIn Live sessions
   • YouTube live Q&A sessions

💼 Professional Networking Tips:

Before Event:
• Research speakers/attendees
• Prepare introduction (30 seconds)
• Have questions ready
• Dress appropriately

During Event:
• Be genuinely curious
• Ask thoughtful questions
• Exchange contact info
• Take notes

After Event:
• Send LinkedIn connection request within 24 hours
• Personalized message mentioning the event
• Share any promised resources
• Follow up in 1-2 weeks

📧 Networking Message Template:

"Hi [Name],

It was great meeting you at [Event Name]. I really enjoyed our conversation about [specific topic].

I'm a [year] year student at [college] studying [field]. I'm particularly interested in [specific interest related to their work].

Would you be open to a quick 15-minute virtual coffee chat? I'd love to learn more about your experience in [their field].

Thank you for your time!

Best regards,
[Your Name]"

🎯 How to Build Your Network:

1. Start with College:
   • Connect with seniors (1-2 years ahead)
   • Join clubs related to your interest
   • Participate in committees
   • Volunteer for events

2. Reach Out Online:
   • 5 LinkedIn connections per week
   • Comment on posts of professionals
   • Share your learnings/projects
   • Join relevant discussions

3. Maintain Relationships:
   • Regular check-ins (quarterly)
   • Share useful content
   • Offer help when possible
   • Remember birthdays/work anniversaries

⚠️ Networking Don'ts:
• Don't ask for jobs immediately
• Don't send generic messages
• Don't be pushy
• Don't forget to follow up
• Don't only reach out when you need something
"""

    return opportunities

# =============================================================================
# COMMUNICATION COACH AGENT TOOLS
# =============================================================================

@tool
def generate_email_template(purpose: str, recipient: str) -> str:
    """
    Use this tool when the user wants help writing any professional email or
    message. This includes emails to professors, HR managers, college admin,
    for leave applications, internship applications, or scholarship requests.

    Args:
        purpose: What the email is for. Examples: query, leave, internship application,
                 scholarship application, doubt, recommendation letter
        recipient: Who is receiving the email. Examples: professor, HR, admin,
                   scholarship committee
    """
    templates = {
        "professor_query": """
📧 EMAIL TO PROFESSOR - DOUBT/QUERY

Subject: Query Regarding [Topic/Subject Name] - [Your Name], [Class/Roll No]

Dear Prof. [Professor's Last Name],

I hope this email finds you well.

I am [Your Full Name], a student in your [Course Code - Course Name] class ([Day and Time of class, e.g., Monday 10 AM section]).

I have been working on [assignment/topic/concept] and have a question regarding [specific topic]. [Briefly explain your doubt in 2-3 sentences].

I have tried [mention what you've already attempted - checked textbook, asked classmates, etc.], but I'm still unclear about [specific confusion].

Would it be possible to clarify this during your office hours, or could you spare a few minutes after class? I want to ensure I understand this concept correctly before the [upcoming exam/assignment deadline].

Thank you for your time and guidance.

Best regards,
[Your Full Name]
[Roll Number]
[Department/Year]
[Contact Number]

---

💡 Tips for Emailing Professors:
• Use clear, specific subject line
• Be respectful and professional
• Be concise but clear
• Proofread before sending
• Send during business hours
• Wait 2-3 days before follow-up
• Use college email ID
""",
        "professor_leave": """
📧 EMAIL TO PROFESSOR - LEAVE APPLICATION

Subject: Leave Application for [Dates] - [Your Name], [Roll No]

Dear Prof. [Professor's Last Name],

I am writing to inform you that I will be unable to attend your [Course Name] classes from [Start Date] to [End Date] due to [brief reason - medical emergency/family function/etc.].

I understand that during this period, we will be covering [topic if you know, or say "important coursework"]. I assure you that I will:
• Obtain notes from my classmates
• Complete any assignments during this period
• Cover the missed topics on my own

[If applicable: I have attached the medical certificate/supporting document for your reference.]

I request you to kindly grant me leave for these dates. I will ensure that my absence does not affect my academic performance.

Thank you for your understanding.

Respectfully,
[Your Full Name]
[Roll Number]
[Department/Year]
[Contact Number]

Attachment: [If any]

---

💡 Important Points:
• Apply in advance when possible
• Attach proof if available
• Show commitment to catch up
• Keep it brief and honest
• Follow college leave policy
""",
        "internship_application": """
📧 EMAIL FOR INTERNSHIP APPLICATION

Subject: Application for [Internship Position] - [Your Name], [College Name]

Dear Hiring Manager / [Name if you know],

I am writing to express my strong interest in the [Internship Position] at [Company Name], as advertised on [where you found it].

I am currently a [Year, e.g., third-year] student pursuing [Your Degree] at [Your College]. I am particularly drawn to this opportunity because [mention 1-2 specific reasons related to the company/role].

My relevant experience includes:
• [Skill/Project 1]: [Brief description and impact]
• [Skill/Project 2]: [Brief description and impact]
• [Relevant coursework or achievement]

I have attached my resume for your review. I am particularly excited about [mention specific project/product of the company] and would love the opportunity to contribute to [specific team/project].

I am available for an interview at your convenience and can start the internship from [available start date].

Thank you for considering my application. I look forward to the opportunity to discuss how I can contribute to your team.

Best regards,
[Your Full Name]
[Contact Number]
[Email Address]
[LinkedIn Profile URL]

Attachments: Resume, Portfolio (if applicable)

---

💡 Application Tips:
• Research the company thoroughly
• Customize for each application
• Highlight relevant skills only
• Keep it to 3-4 paragraphs
• Proofread multiple times
• Send during business hours
• Follow up after 1 week
""",
        "admin_query": """
📧 EMAIL TO COLLEGE ADMINISTRATION

Subject: Query Regarding [Specific Issue] - [Your Name], [Roll No]

Dear Sir/Madam,

I am [Your Full Name], a [Year] year student of [Department/Course] (Roll No: [Your Roll Number]).

I am writing to inquire about [specific issue - fee payment/certificate/document/hostel/etc.].

[Clearly explain your query in 2-3 sentences. Include relevant details like dates, reference numbers, previous communications if any.]

I would appreciate if you could:
• [Specific action you need - clarify process/provide information/resolve issue]
• [Any additional request]

I have attached [relevant documents if any] for your reference.

Please let me know if you need any additional information from my side.

Thank you for your assistance.

Yours sincerely,
[Your Full Name]
[Roll Number]
[Department/Year]
[Contact Number]
[Email Address]

---

💡 Tips for Admin Emails:
• Be very specific about your issue
• Include roll number always
• Attach relevant documents
• Be polite and patient
• Visit office if no reply in 3-4 days
""",
        "scholarship_application": """
📧 EMAIL FOR SCHOLARSHIP APPLICATION

Subject: Application for [Scholarship Name] - [Your Name]

Dear Scholarship Committee / Sir/Madam,

I am writing to apply for the [Scholarship Name] offered by [Organization/College]. I am currently a [Year] year student pursuing [Degree] at [College Name].

I am from a [brief background - first-generation college student/economically weaker section/etc.] and this scholarship would significantly help me continue my education.

Academic Background:
• Current CGPA: [Your CGPA]
• Academic achievements: [Key achievements]
• Extracurricular activities: [Brief mention]

Financial Need:
[Briefly and honestly explain your financial situation - family income, siblings' education, expenses, etc. Keep it factual and dignified.]

How This Scholarship Will Help:
This scholarship will enable me to [continue education without financial stress/focus on academics/pursue additional certifications/etc.]. I am committed to utilizing this opportunity to excel academically and contribute positively to society.

I have attached all required documents as per the application guidelines:
• Income certificate
• Fee receipts
• Marksheets
• [Other documents]

Thank you for considering my application. I would be happy to provide any additional information needed.

Respectfully,
[Your Full Name]
[Roll Number]
[Department/Course/Year]
[Contact Number]
[Email Address]

Attachments: [List all attached documents]

---

💡 Scholarship Application Tips:
• Be honest about financial need
• Emphasize academic commitment
• Show gratitude
• Attach ALL required documents
• Proofread carefully
• Submit before deadline
• Follow up on application status
"""
    }

    purpose_lower = purpose.lower()
    recipient_lower = recipient.lower()

    # Improved matching logic
    if "professor" in recipient_lower or "teacher" in recipient_lower or "faculty" in recipient_lower:
        if any(w in purpose_lower for w in ["leave", "absent", "miss", "sick"]):
            return templates["professor_leave"]
        else:
            return templates["professor_query"]
    elif any(w in purpose_lower for w in ["internship", "job", "work", "hr", "company", "hiring"]):
        return templates["internship_application"]
    elif any(w in purpose_lower for w in ["scholarship", "financial", "aid", "stipend", "fee waiver"]):
        return templates["scholarship_application"]
    elif any(w in recipient_lower for w in ["admin", "office", "registrar", "department"]):
        return templates["admin_query"]

    # Default: show all options
    return """
📧 PROFESSIONAL EMAIL TEMPLATES AVAILABLE:

I can help you write emails for:

1. **To Professors:**
   - Asking doubts/queries about coursework
   - Requesting leave (medical or personal)
   - Requesting recommendation letters
   - Discussing grades

2. **For Opportunities:**
   - Internship applications to companies
   - Scholarship applications
   - Research assistant positions

3. **To College Admin:**
   - Fee-related queries
   - Document requests (bonafide, transcripts)
   - Hostel issues
   - General complaints/queries

Please tell me:
• **Who** are you emailing? (professor / HR / admin)
• **What** is the email for? (leave / query / internship / scholarship)

I'll generate the perfect template for you! 📝
"""

@tool
def linkedin_profile_guide(section: str) -> str:
    """
    Use this tool when the user asks about LinkedIn, building a professional
    online presence, improving their profile, or how to present themselves
    to recruiters and employers online.

    Args:
        section: Which part of LinkedIn to help with. Examples: headline, summary,
                 experience, skills, general, photo, connections, posts
    """
    guides = {
        "headline": """
💼 LINKEDIN HEADLINE GUIDE

Your headline appears everywhere on LinkedIn - make it count!

✅ Good Headline Structure:
[Current Role/Status] | [Key Skills] | [Goal/Interest]

Examples for Students:

1. Computer Science Student:
"Computer Science Student | Python, Web Dev | Seeking SDE Internship"

2. Business Student:
"MBA Student | Marketing & Analytics | Aspiring Brand Manager"

3. First-Year Student:
"Engineering Student @ [College] | Learning Data Science | Open to Opportunities"

4. Final Year:
"Final Year CSE | Full Stack Developer | Actively Seeking SDE Roles"

❌ Avoid:
• "Student at [College]" (too generic)
• Just your degree name
• Emojis overload
• Unprofessional language

💡 Pro Tips:
• Update headline every 3-6 months
• Include keywords recruiters search for
• Mention if actively looking for opportunities
• Keep it under 120 characters
""",
        "summary": """
📝 LINKEDIN SUMMARY/ABOUT GUIDE

Your summary is your elevator pitch - make it compelling!

✅ Structure (4-Paragraph Format):

Paragraph 1 - Who You Are:
"I'm a third-year Computer Science student at [College] with a passion for building scalable web applications."

Paragraph 2 - What You've Done (use bullet points):
"My experience includes:
• Built 5+ web applications using React and Node.js
• Completed internship at [Company] on [Project]
• Led technical team of college fest with 500+ participants"

Paragraph 3 - Current Focus:
"Currently, I'm diving deep into cloud computing. I'm working on [Current Project] and learning AWS."

Paragraph 4 - Call to Action:
"I'm actively seeking SDE internship opportunities for Summer 2025. Connect if you'd like to discuss tech, projects, or opportunities!"

💡 Pro Tips:
• Write in first person ("I" not "He/She")
• Keep paragraphs short (2-3 lines)
• Include keywords for searchability
• Show personality, not just achievements
• Add contact info at the end
• Update every 6 months
""",
        "experience": """
💼 LINKEDIN EXPERIENCE SECTION GUIDE

Even without formal experience, you have valuable skills to showcase!

✅ Types of Experience to Include:
1. Internships (formal or informal)
2. College Projects (list them as experience!)
3. Freelance Work (even small gigs)
4. Club Leadership (technical clubs, committees)
5. Volunteer Work
6. Online Course Capstone Projects

📝 Example - College Project Entry:

**Software Development Project**
[Your College Name] - Academic Project
June 2024 - August 2024

• Built e-commerce web app using MERN stack, tested with 100+ users
• Integrated secure payment gateway and user authentication
• Collaborated with team of 4 using Git for version control
Skills: React.js · Node.js · MongoDB · Git

📝 Example - Club Position Entry:

**Technical Lead**
[College Technical Club] | August 2023 - Present

• Led 15-member team organizing workshops and hackathons
• Conducted 5 workshops on web development for 200+ students
• Grew club social media following by 150%
Skills: Leadership · Event Management · Technical Training

💡 Pro Tips:
• Use action verbs (Developed, Led, Built, Managed, Increased)
• Include numbers/metrics wherever possible
• Add screenshots or links as media attachments
• List 4-5 skills at the end of each entry
• Keep bullet points to 1-2 lines each

❌ Common Mistakes:
• Writing paragraphs instead of bullet points
• No measurable outcomes
• Listing duties instead of achievements
• Grammar/spelling errors
"""
    }

    section_lower = section.lower()
    for key in guides:
        if key in section_lower:
            return guides[key]

    return """
💼 LINKEDIN PROFILE OPTIMIZATION GUIDE

I can give detailed help for any of these sections:

**Core Sections:**
• **Headline** - First impression, shows up in every search result
• **Summary/About** - Your story, goals, and elevator pitch
• **Experience** - Internships, projects, club roles (all count!)
• **Skills** - Technical + soft skills (aim for 10+)
• **Education** - College, courses, certifications

**Quick Wins Checklist:**
✅ Professional photo (clear face, light background)
✅ Custom headline (NOT just "Student at [College]")
✅ Compelling About section (4 paragraphs)
✅ At least 3 Experience entries (projects count!)
✅ 10+ skills added and endorsed
✅ Custom LinkedIn URL (linkedin.com/in/yourname)
✅ "Open to Work" badge turned on

📈 Growth Tips:
• Post once a week (project updates, learnings)
• Connect with 5 new people every week
• Comment thoughtfully on posts in your field
• Join 3-5 LinkedIn groups related to your interest
• Follow companies you want to work at
• Ask seniors for skill endorsements

Which section would you like detailed guidance on?
(headline / summary / experience / skills)
"""

# =============================================================================
# LEARNING ROADMAP AGENT TOOL
# =============================================================================

@tool
def generate_learning_roadmap(topic: str, current_level: str = "beginner") -> str:
    """
    Use this tool when the user wants to LEARN something new, asks for a
    roadmap, study plan, or how to get started with any skill or subject.
    This includes programming languages, frameworks, domains like data science,
    or any career skill.

    ALWAYS use this tool for requests like:
    - "I want to learn [topic]"
    - "How do I start with [topic]"
    - "Teach me [topic]"
    - "Roadmap for [topic]"
    - "How to become a [role]"

    Args:
        topic: The subject to learn e.g. Python, web development, data science,
               machine learning, graphic design, digital marketing
        current_level: beginner, intermediate, or advanced (default: beginner)
    """
    try:
        query = f"complete learning roadmap {topic} {current_level} free resources 2025"

        tavily_tool = TavilySearchResults(
            max_results=7,
            search_depth="advanced",
            include_answer=True
        )

        results = tavily_tool.invoke({"query": query})

        roadmap = f"""
📚 COMPLETE LEARNING ROADMAP: {topic.upper()}
Level: {current_level.capitalize()}

"""

        topic_lower = topic.lower()

        if any(w in topic_lower for w in ["web", "frontend", "front-end", "html", "css", "react"]):
            roadmap += """
🗺️ PHASE-WISE LEARNING PATH:

📍 PHASE 1: Foundations (2-3 months)
Week 1-4: HTML & CSS Basics
• Learn HTML tags, forms, semantic HTML
• CSS styling, flexbox, grid layouts
• Build 3-5 static web pages

Week 5-8: JavaScript Fundamentals
• Variables, data types, functions
• DOM manipulation and events
• ES6+ features (arrow functions, promises)
• Build interactive web pages

Week 9-12: Version Control & Responsive Design
• Git and GitHub basics
• Responsive design principles
• CSS frameworks (Bootstrap or Tailwind)
• Build a responsive portfolio website

📍 PHASE 2: Modern Frontend (3-4 months)
Month 4-5: React.js
• Components, props, state management
• Hooks (useState, useEffect, custom hooks)
• React Router for navigation
• Build 3-4 React applications

Month 6: API Integration & State Management
• Fetch API and Axios
• Context API or Redux basics
• Authentication fundamentals
• Build apps that consume real APIs

Month 7: Advanced Concepts
• Performance optimization
• Basic testing (Jest)
• TypeScript introduction
• Build 2 polished portfolio projects

📍 PHASE 3: Professional Skills (2 months)
• Build complete portfolio with 5+ projects
• Learn deployment (Vercel, Netlify - both free)
• Code quality and best practices
• Contribute to open source

🎯 TOTAL TIMELINE: 6-9 months with daily practice
"""
        elif any(w in topic_lower for w in ["python", "programming"]):
            roadmap += """
🗺️ PHASE-WISE LEARNING PATH:

📍 PHASE 1: Python Basics (1-2 months)
Week 1-2: Getting Started
• Installation and setup (free)
• Variables, data types, operators
• Input/output, conditionals
• Practice: 20-30 basic problems on HackerRank

Week 3-4: Control Flow
• If-else, for loops, while loops
• Break, continue, nested loops
• Practice: 30-40 problems

Week 5-6: Functions & Data Structures
• Functions, parameters, return values
• Lists, tuples, dictionaries, sets
• List comprehensions
• Build 3 mini projects

Week 7-8: Object-Oriented Programming
• Classes and objects
• Inheritance and polymorphism
• Build 2-3 OOP projects

📍 PHASE 2: Intermediate Python (2-3 months)
• File handling (read/write CSV, JSON)
• Exception handling
• Modules and packages (pip)
• Working with APIs (requests library)
• Database basics (SQLite)
• Build 3-4 practical projects

📍 PHASE 3: Pick Your Specialization (3-4 months)

Option A - Web Development:
→ Learn Flask or Django framework
→ HTML/CSS/JavaScript basics
→ Build full-stack web applications

Option B - Data Science:
→ NumPy and Pandas for data analysis
→ Matplotlib and Seaborn for charts
→ Work on real-world datasets (Kaggle)

Option C - Automation:
→ Selenium for browser automation
→ BeautifulSoup for web scraping
→ Build tools that save you time

🎯 TOTAL TIMELINE: 6-9 months to job-ready
"""
        elif any(w in topic_lower for w in ["data science", "machine learning", "ml", "ai", "artificial intelligence"]):
            roadmap += """
🗺️ PHASE-WISE LEARNING PATH:

📍 PHASE 1: Foundations (2-3 months)
Month 1: Python Programming
• Python basics and syntax
• Data structures (lists, dicts, sets)
• Functions, modules, file handling
• Practice coding problems daily

Month 2: Math & Statistics (don't skip!)
• Descriptive statistics (mean, median, std)
• Probability basics
• Linear algebra essentials
• Use Khan Academy (free)

Month 3: Data Manipulation
• NumPy for numerical computing
• Pandas for data cleaning & analysis
• Work with CSV and Excel files
• Complete 3 data cleaning projects

📍 PHASE 2: Data Analysis (2-3 months)
Month 4: Data Visualization
• Matplotlib for basic plots
• Seaborn for statistical charts
• Build dashboards
• Tell stories with data (key skill!)

Month 5: SQL & Databases
• SELECT, WHERE, JOIN, GROUP BY
• Practice on SQLZoo or Mode Analytics
• Combine SQL with Python (pandas + SQL)

Month 6: Exploratory Data Analysis (EDA)
• Full EDA workflow on real datasets
• Finding patterns and outliers
• Statistical hypothesis testing
• Work on 2-3 Kaggle datasets

📍 PHASE 3: Machine Learning (3-4 months)
Month 7-8: ML Fundamentals
• Scikit-learn library
• Supervised: regression, classification
• Unsupervised: clustering
• Model evaluation (accuracy, F1, AUC)

Month 9-10: ML Projects
• Feature engineering techniques
• Model tuning (GridSearchCV)
• Ensemble methods (Random Forest, XGBoost)
• Build and document 3-4 ML projects

📍 PHASE 4: Specialization (2-3 months)
Choose one area:
• Deep Learning (TensorFlow/PyTorch)
• NLP - Natural Language Processing
• Computer Vision
• Time Series Forecasting

🎯 TOTAL TIMELINE: 10-12 months to job-ready
"""
        else:
            roadmap += f"""
🗺️ LEARNING PATH FOR {topic.upper()}:

📍 PHASE 1: Foundations (25% of journey)
• Understand core concepts and terminology
• Learn fundamental principles
• Follow one structured course completely
• Build conceptual clarity before moving forward

📍 PHASE 2: Core Skills (35% of journey)
• Deep dive into key topics
• Hands-on practice and exercises
• Learn industry-standard tools
• Build 2-3 beginner projects

📍 PHASE 3: Advanced Concepts (25% of journey)
• Master complex topics
• Work on real-world projects
• Learn professional best practices
• Build portfolio-worthy projects

📍 PHASE 4: Professional Ready (15% of journey)
• Polish your projects and portfolio
• Network and share your work online
• Apply for opportunities
• Keep learning continuously

"""

        roadmap += """
🔗 BEST FREE LEARNING RESOURCES:

**📺 YouTube Channels (Free & Excellent):**
• freeCodeCamp - Full courses, 100% free
• Traversy Media - Practical project tutorials
• Programming with Mosh - Clear explanations
• Corey Schafer - Python and more
• The Net Ninja - Web development

**💻 Interactive Platforms (Free):**
• freeCodeCamp.org - Certifications at zero cost
• The Odin Project - Full web dev curriculum
• Codecademy (free tier) - Interactive coding
• CS50 by Harvard (edX) - Best intro CS course
• Kaggle Learn - Data science mini courses

**🏋️ Practice Platforms:**
• LeetCode - Coding interview problems
• HackerRank - Beginner-friendly challenges
• Codewars - Fun coding exercises
• Kaggle - Real datasets and competitions

**📖 Documentation & Reference:**
• MDN Web Docs - Everything web
• GeeksforGeeks - Concepts and problems
• Real Python - Python tutorials
• Official documentation of any tool

**👥 Communities to Join:**
• Stack Overflow - Ask and answer questions
• Reddit: r/learnprogramming, r/datascience
• Discord servers for your topic
• Dev.to - Developer articles and blogs
"""

        if results and isinstance(results, list):
            roadmap += "\n**🔍 Current Resources Found Online:**\n\n"
            for i, result in enumerate(results[:4], 1):
                if isinstance(result, dict):
                    title = result.get('title', 'Resource')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    roadmap += f"{i}. **{title}**\n   {content[:120]}...\n   🔗 {url}\n\n"

        roadmap += """
📅 SUGGESTED WEEKLY SCHEDULE:

For Students (15-20 hrs/week):
• Mon-Fri: 2 hours/day
  - 1 hour: Learn new concept (video/reading)
  - 1 hour: Practice/code it yourself
• Saturday: 4-5 hours (build project)
• Sunday: 2-3 hours (review + plan next week)

💡 THE MOST IMPORTANT RULES:

✅ DO:
• Code every single day (even 30 minutes counts)
• Build projects — tutorials alone won't get you hired
• Push everything to GitHub (your public portfolio)
• Ask questions on Stack Overflow and communities
• Document your learning (blog/LinkedIn posts)

❌ DON'T:
• Switch resources every week (stick to one!)
• Skip fundamentals to jump to advanced topics
• Watch tutorials without coding along
• Compare your pace to others online

🏆 KEY MILESTONES TO CELEBRATE:
□ Finish your first tutorial
□ Build your first project from scratch
□ Push first project to GitHub
□ Solve 50 practice problems
□ Build a portfolio website
□ Apply for first opportunity

Remember: 30 minutes every day beats 5 hours once a week!
Consistency is your superpower. 🚀

Ask me about any specific phase, project ideas, or resources! 💬
"""

        return roadmap

    except Exception as e:
        return f"""
📚 LEARNING ROADMAP: {topic.upper()}

🗺️ GENERAL LEARNING STRUCTURE:

1. **Start with Basics** — Don't skip foundations!
2. **Practice Daily** — Consistency beats intensity
3. **Build Projects** — Apply what you learn immediately
4. **Join Communities** — Learn faster together
5. **Share Your Work** — GitHub + LinkedIn

🔗 TOP FREE RESOURCES:
• **YouTube**: freeCodeCamp, Traversy Media, Corey Schafer
• **Websites**: freeCodeCamp.org, The Odin Project, Kaggle
• **Practice**: LeetCode, HackerRank, Codewars
• **Reference**: GeeksforGeeks, Official docs, MDN

⏰ REALISTIC TIMELINE: 6-12 months to job-ready with consistent effort

Ask me about any specific aspect of learning {topic}! 🎯
"""

# =============================================================================
# CREATE MULTI-AGENT SYSTEM
# =============================================================================

def create_agent():
    """Initialize and return the AI mentor multi-agent system."""

    # FIX 1: Lower temperature for more precise, relevant answers
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=2048,
        timeout=60,
        max_retries=2
    )

    tavily_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False
    )

    tools = [
        get_current_datetime,
        get_college_deadlines,
        explain_college_process,
        search_internships,
        find_scholarships,
        networking_opportunities,
        generate_email_template,
        linkedin_profile_guide,
        generate_learning_roadmap,
        tavily_tool
    ]

    # FIX 2: Much more precise system prompt with explicit tool routing rules
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI Mentor for first-generation college students in India. You are warm, practical, and encouraging.

=== CRITICAL TOOL USAGE RULES — FOLLOW THESE EXACTLY ===

You MUST call a tool before responding whenever the user's message matches these categories:

1. LEARNING REQUEST → call generate_learning_roadmap
   Triggers: "I want to learn", "how to learn", "teach me", "roadmap for", "how to become", "get started with", "study plan for"
   Example: "I want to learn web development" → call generate_learning_roadmap(topic="web development", current_level="beginner")

2. INTERNSHIP REQUEST → call search_internships
   Triggers: "find internship", "internship for", "summer internship", "winter internship", "work experience", "placement"
   Example: "Find me internships for CSE 3rd year" → call search_internships(field="computer science", year="3rd year")

3. SCHOLARSHIP REQUEST → call find_scholarships
   Triggers: "scholarship", "financial aid", "fee waiver", "money for education", "stipend"
   Example: "What scholarships can I apply for?" → call find_scholarships(category="general")

4. EMAIL WRITING REQUEST → call generate_email_template
   Triggers: "write email", "email to professor", "email for leave", "help me write", "draft email", "email template"
   Example: "Write an email to professor for leave" → call generate_email_template(purpose="leave", recipient="professor")

5. COLLEGE PROCESS QUESTION → call explain_college_process
   Triggers: "how does", "explain", "what is the process", "how to register", "how hostel", "attendance rules", "grading system", "CGPA"
   Example: "How do I register for courses?" → call explain_college_process(process_name="registration")

6. DEADLINE/DATE QUESTION → call get_college_deadlines
   Triggers: "when is", "deadline for", "timeline for", "when should I apply", "important dates"
   Example: "When are scholarship deadlines?" → call get_college_deadlines(event_type="scholarship")

7. LINKEDIN/PROFILE QUESTION → call linkedin_profile_guide
   Triggers: "linkedin", "profile", "online presence", "how to build profile", "headline", "summary"
   Example: "How to optimize my LinkedIn?" → call linkedin_profile_guide(section="general")

8. NETWORKING QUESTION → call networking_opportunities
   Triggers: "networking", "meet professionals", "communities", "events", "connect with people"
   Example: "How do I network in tech?" → call networking_opportunities(interest="tech")

=== RESPONSE RULES ===

- ALWAYS call the appropriate tool first. Do NOT answer from memory when a tool exists for that topic.
- After the tool returns data, present it clearly to the student.
- Add 1-2 sentences of your own encouragement after the tool response.
- If the user provides more context (like their year, field, or category), pass it as tool arguments.
- If essential information is missing (e.g., field of study for internship search), make a reasonable assumption and tell the student what you assumed.
- NEVER make up scholarship names, company programs, or deadlines — use tool data.
- Keep your own added commentary short. The tool output is the main answer.
- Use simple language. Avoid jargon. This may be the first time the student is navigating these systems.

=== TONE ===
- Warm and encouraging, like a helpful senior student
- Never condescending
- Celebrate small wins and normalize asking for help
- If a student seems stressed or confused, acknowledge that before giving information"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    # FIX 3: Tighter iteration limits to prevent wandering
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=90,
        return_intermediate_steps=False
    )

    return agent_executor

def chat(user_input: str, agent_executor):
    """Process user input and maintain chat history."""
    global chat_history

    try:
        if chat_history is None:
            chat_history = []

        # FIX 4: Robust chat history conversion with validation
        formatted_history = []
        for msg in chat_history:
            if not (isinstance(msg, tuple) and len(msg) == 2):
                continue
            role, content = msg
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "human":
                formatted_history.append(HumanMessage(content=content))
            elif role == "assistant":
                formatted_history.append(AIMessage(content=content))

        if agent_executor is None:
            agent_executor = create_agent()

        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": formatted_history
        })

        # FIX 5: Cleaner output extraction
        if isinstance(response, dict):
            output = response.get("output", "").strip()
        elif hasattr(response, "output"):
            output = str(response.output).strip()
        else:
            output = str(response).strip()

        if not output:
            output = "I'm here to help! Could you rephrase your question?"

        # Save to history only if we got a real response
        chat_history.append(("human", user_input))
        chat_history.append(("assistant", output))

        # Keep history bounded to last 10 exchanges (20 messages)
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        return output

    except Exception as e:
        print(f"Chat error: {str(e)}")
        return "I ran into a technical issue. Could you please try asking again?"


if __name__ == "__main__":
    print("=" * 70)
    print("🎓 AI MENTOR FOR FIRST-GENERATION COLLEGE STUDENTS")
    print("=" * 70)
    print("\n📚 Your personal guide for college success!\n")
    print("I can help you with:")
    print("• College processes and deadlines")
    print("• Finding internships and scholarships")
    print("• Writing professional emails")
    print("• Building your LinkedIn profile")
    print("• Learning new skills (with complete roadmaps!)")
    print("• Networking and career guidance")
    print("\n" + "=" * 70)

    print("\n🔑 Loading API keys...")

    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️  GROQ_API_KEY not found!")
        print("\nCreate a .env file with:")
        print("GROQ_API_KEY=gsk-your-key-here")
        print("TAVILY_API_KEY=tvly-your-key-here")
        sys.exit(1)

    if not os.getenv("TAVILY_API_KEY"):
        print("\n⚠️  TAVILY_API_KEY not found!")
        sys.exit(1)

    print("✅ API keys loaded!")
    print("\n🤖 Initializing AI Mentor...")

    try:
        agent_executor = create_agent()
        print("✅ AI Mentor is ready!\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize: {str(e)}")
        sys.exit(1)

    print("💬 Start chatting! (type 'quit' to exit, 'clear' to reset)\n")
    print("Examples:")
    print("• 'How do I register for courses?'")
    print("• 'Find me internships in data science'")
    print("• 'I want to learn web development from scratch'")
    print("• 'Help me write an email to my professor for leave'")
    print("• 'How to apply for scholarships?'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Keep learning and growing! Good luck! 🌟")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Remember: You've got this! Keep pushing forward! 🌟")
            break

        if user_input.lower() == 'clear':
            chat_history.clear()
            print("\n✅ Chat history cleared!\n")
            continue

        try:
            response = chat(user_input, agent_executor)
            print(f"\n🎓 AI Mentor: {response}\n")
        except KeyboardInterrupt:
            print("\n\nType 'quit' to exit.\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
