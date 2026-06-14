"""
AI Interview Engine
"""
from groq import Groq

GROQ_API_KEY = "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"  # paste your real key
client = Groq(api_key=GROQ_API_KEY)


def parse_resume_and_generate_questions(resume_text: str, job_role: str) -> list:
    import re
    # Clean garbled text
    clean_resume = re.sub(r'[^\x20-\x7E\s]', '', resume_text)
    clean_resume = re.sub(r'\s+', ' ', clean_resume).strip()
    
    # If resume text is too garbled, use just the job role
    if len(clean_resume) < 50:
        clean_resume = f"Candidate applying for {job_role} position."

    prompt = f"""Generate exactly 5 interview questions for a {job_role} position.

Candidate background: {clean_resume[:1500]}

Return ONLY a numbered list:
1. First question
2. Second question
3. Third question
4. Fourth question
5. Fifth question"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.7,
    )
    text = response.choices[0].message.content.strip()
    questions = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^[\d]+[.)]\s*', '', line).strip()
        if len(cleaned) > 15:
            questions.append(cleaned)
    if not questions:
        questions = [text]
    return questions[:5]


def generate_coding_questions(job_role: str, language: str) -> list:
    prompt = f"""
You are a technical interviewer. Generate exactly 2 coding problems for a {job_role} position.
Language: {language}

For each problem provide:
- Problem title
- Problem description
- Example input and output
- Constraints

Format each problem clearly. Number them 1 and 2.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.7,
    )
    text = response.choices[0].message.content
    problems = []
    parts = text.split('\n\n')
    current = []
    for part in parts:
        if part.strip():
            current.append(part.strip())
            if len('\n\n'.join(current)) > 100:
                problems.append('\n\n'.join(current))
                current = []
                if len(problems) == 2:
                    break
    if current and len(problems) < 2:
        problems.append('\n\n'.join(current))
    return problems[:2] if problems else [text]


def evaluate_answer(question: str, answer: str, job_role: str) -> dict:
    prompt = f"""
You are an expert interviewer evaluating a candidate's answer.

Job Role: {job_role}
Question: {question}
Candidate's Answer: {answer}

Evaluate the answer and provide:
1. Score out of 10
2. What was good about the answer
3. What could be improved
4. Overall feedback

Be concise and constructive. Format as:
SCORE: X/10
GOOD: ...
IMPROVE: ...
FEEDBACK: ...
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.5,
    )
    text = response.choices[0].message.content
    result = {"score": 0, "good": "", "improve": "", "feedback": "", "raw": text}
    for line in text.split('\n'):
        if line.startswith('SCORE:'):
            try:
                result["score"] = int(line.split(':')[1].strip().split('/')[0])
            except:
                result["score"] = 5
        elif line.startswith('GOOD:'):
            result["good"] = line.replace('GOOD:', '').strip()
        elif line.startswith('IMPROVE:'):
            result["improve"] = line.replace('IMPROVE:', '').strip()
        elif line.startswith('FEEDBACK:'):
            result["feedback"] = line.replace('FEEDBACK:', '').strip()
    return result


def evaluate_code(problem: str, code: str, language: str) -> dict:
    prompt = f"""
You are an expert code reviewer. Evaluate the following code solution.

Problem: {problem}
Language: {language}
Code submitted:
```{language.lower()}
{code}
```

Evaluate and provide:
1. Is the code correct? (Yes/No)
2. Score out of 10
3. Time complexity
4. Space complexity
5. What is good about the code
6. Issues or bugs if any
7. Hints for improvement
8. Optimal solution approach

Format as:
CORRECT: Yes/No
SCORE: X/10
TIME: O(...)
SPACE: O(...)
GOOD: ...
ISSUES: ...
HINTS: ...
OPTIMAL: ...
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )
    text = response.choices[0].message.content
    result = {
        "correct": False, "score": 0, "time": "", "space": "",
        "good": "", "issues": "", "hints": "", "optimal": "", "raw": text
    }
    for line in text.split('\n'):
        if line.startswith('CORRECT:'):
            result["correct"] = 'yes' in line.lower()
        elif line.startswith('SCORE:'):
            try:
                result["score"] = int(line.split(':')[1].strip().split('/')[0])
            except:
                result["score"] = 5
        elif line.startswith('TIME:'):
            result["time"] = line.replace('TIME:', '').strip()
        elif line.startswith('SPACE:'):
            result["space"] = line.replace('SPACE:', '').strip()
        elif line.startswith('GOOD:'):
            result["good"] = line.replace('GOOD:', '').strip()
        elif line.startswith('ISSUES:'):
            result["issues"] = line.replace('ISSUES:', '').strip()
        elif line.startswith('HINTS:'):
            result["hints"] = line.replace('HINTS:', '').strip()
        elif line.startswith('OPTIMAL:'):
            result["optimal"] = line.replace('OPTIMAL:', '').strip()
    return result


def generate_final_report(
    job_role: str,
    behavioral_scores: list,
    coding_scores: list,
    candidate_name: str = "Candidate"
) -> str:
    avg_behavioral = sum(behavioral_scores) / len(behavioral_scores) if behavioral_scores else 0
    avg_coding = sum(coding_scores) / len(coding_scores) if coding_scores else 0
    overall = (avg_behavioral * 0.6 + avg_coding * 0.4)

    if overall >= 8:
        verdict = "STRONGLY RECOMMENDED ✅"
    elif overall >= 6:
        verdict = "RECOMMENDED ✅"
    elif overall >= 4:
        verdict = "MAYBE ⚠️"
    else:
        verdict = "NOT RECOMMENDED ❌"

    return f"""
INTERVIEW REPORT
================
Candidate: {candidate_name}
Role: {job_role}

SCORES:
- Behavioral: {avg_behavioral:.1f}/10
- Coding: {avg_coding:.1f}/10
- Overall: {overall:.1f}/10

VERDICT: {verdict}
"""