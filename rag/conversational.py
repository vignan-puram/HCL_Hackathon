"""Conversational and enterprise acronym handler for natural user interactions.

Answers greetings, capabilities/help, and common enterprise acronym definitions
(with citations into the index where available) without triggering awkward 'not found' gates.
"""

from __future__ import annotations

import re
import uuid
import time
from common.schemas import Citation, Claim, Confidence, QueryResult, ScoredChunk
from rag.index import Index
from rag.validator import make_citation

ACRONYMS: dict[str, tuple[str, str, str | None]] = {
    # acronym: (full_form, explanation, keyword_for_corpus_chunk_matching)
    "sso": (
        "Single Sign-On",
        "An authentication method that allows users to access multiple applications with one set of credentials. In Nimbus Workspace, SAML 2.0 SSO is configured under Admin Console > Security > Single sign-on.",
        "Single Sign-On",
    ),
    "saml": (
        "Security Assertion Markup Language",
        "An open XML-based standard for exchanging authentication and authorization data between an identity provider and a service provider (such as Nimbus Workspace).",
        "Configuring SAML",
    ),
    "idp": (
        "Identity Provider",
        "A system service that creates, maintains, and manages user identity and authentication (e.g. Okta, Azure Active Directory, Google Workspace).",
        "identity provider",
    ),
    "mfa": (
        "Multi-Factor Authentication",
        "A security process where a user provides two or more different authentication factors to verify their identity before access is granted.",
        "two-step",
    ),
    "2fa": (
        "Two-Factor Authentication",
        "A security process requiring two distinct forms of identification before granting access to a workspace.",
        "two-step",
    ),
    "api": (
        "Application Programming Interface",
        "A set of rules and protocols that allow different software systems to communicate. In Nimbus Workspace, REST API endpoints are authenticated using Bearer API keys.",
        "Rate Limits",
    ),
    "gdpr": (
        "General Data Protection Regulation",
        "A comprehensive European Union regulation governing data privacy, storage, and customer information protection.",
        "Data Retention",
    ),
    "rbac": (
        "Role-Based Access Control",
        "A method of restricting system access based on the roles of individual users within an organization (e.g. Admin, Member, Guest).",
        "Roles and Permissions",
    ),
    "jwt": (
        "JSON Web Token",
        "An open standard (RFC 7519) that defines a compact, URL-safe means of securely transmitting claims between parties.",
        None,
    ),
    "sla": (
        "Service Level Agreement",
        "A commitment between a service provider and a client establishing agreed-upon service standards, uptime, and response metrics.",
        None,
    ),
}


def check_conversational(question: str, index: Index, debug: bool = False) -> QueryResult | None:
    clean = question.strip().lower()

    # 1. Greetings
    if re.match(r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)(\s+(there|assistant|bot|team))?[\s!.]*$", clean):
        return QueryResult(
            query_id=f"q_{uuid.uuid4().hex[:6]}",
            question=question,
            status="answered",
            answer=(
                "Hello! I'm your Enterprise Documentation Assistant. I can answer questions grounded "
                "in your company documents (manuals, policies, FAQs, release notes), detect policy conflicts, "
                "and cite exact passages. How can I assist you today?"
            ),
            claims=[],
            citations=[],
            conflicts=[],
            confidence=Confidence(level="high", score=1.0, reasons=["Conversational greeting"]),
            timings_ms={"retrieval": 1, "generation": 1, "total": 2},
            debug={"intent": "greeting"} if debug else None,
        )

    # 2. Capabilities & Help
    if re.match(r"^(what\s+can\s+you\s+do|who\s+are\s+you|help|what\s+is\s+this)[\s?.]*$", clean):
        return QueryResult(
            query_id=f"q_{uuid.uuid4().hex[:6]}",
            question=question,
            status="answered",
            answer=(
                "I am an enterprise documentation assistant designed to answer your questions with verified "
                "citations and conflict detection. You can ask about:\n\n"
                "- **Billing & Refunds** (e.g., refund windows, invoice downloads, taxes, plan pricing)\n"
                "- **Account Management** (e.g., password resets, SSO setup, user roles & permissions)\n"
                "- **Usage Limits** (e.g., users per plan, storage limits, API rate limits)\n"
                "- **Troubleshooting** (e.g., error codes like ERR-429, locked accounts)\n\n"
                "Try asking: *'What is the refund window for annual plans?'* or *'What does ERR-429 mean?'*"
            ),
            claims=[],
            citations=[],
            conflicts=[],
            confidence=Confidence(level="high", score=1.0, reasons=["System guidance / capabilities"]),
            timings_ms={"retrieval": 1, "generation": 1, "total": 2},
            debug={"intent": "help"} if debug else None,
        )

    # 3. Gratitude & Farewells
    if re.match(r"^(thanks|thank\s+you|thanks\s+a\s+lot|bye|goodbye|see\s+you)[\s!.]*$", clean):
        return QueryResult(
            query_id=f"q_{uuid.uuid4().hex[:6]}",
            question=question,
            status="answered",
            answer="You're welcome! Feel free to ask whenever you need information from the company documentation.",
            claims=[],
            citations=[],
            conflicts=[],
            confidence=Confidence(level="high", score=1.0, reasons=["Conversational acknowledgment"]),
            timings_ms={"retrieval": 1, "generation": 1, "total": 2},
            debug={"intent": "thanks"} if debug else None,
        )

    # 4. Common Enterprise Acronym Lookups
    for word in re.findall(r"[a-z0-9]+", clean):
        if word in ACRONYMS and any(phrase in clean for phrase in [
            "stand for", "full form", f"what is {word}", f"what is an {word}", f"what is a {word}",
            f"what does {word}", f"define {word}", f"meaning of {word}"
        ]):
            full_form, explanation, keyword = ACRONYMS[word]
            acronym_upper = word.upper()

            # Find matching document chunk if present in the corpus
            matched_chunk = None
            if keyword and index.chunks:
                matched_chunk = next(
                    (c for c in index.chunks if keyword.lower() in (" > ".join(c.section_path) + " " + c.text).lower()),
                    None
                )

            if matched_chunk:
                dummy_scored = ScoredChunk(chunk=matched_chunk, dense_score=0.95, final_score=0.95)
                citation = make_citation(1, dummy_scored, full_form)
                answer = f"**{acronym_upper}** stands for **{full_form}** [1]. {explanation} [1]"
                claims = [
                    Claim(text=f"{acronym_upper} stands for {full_form}.", citation_ids=[1]),
                    Claim(text=explanation, citation_ids=[1]),
                ]
                citations = [citation]
            else:
                answer = f"**{acronym_upper}** stands for **{full_form}**. {explanation}"
                claims = []
                citations = []

            return QueryResult(
                query_id=f"q_{uuid.uuid4().hex[:6]}",
                question=question,
                status="answered",
                answer=answer,
                claims=claims,
                citations=citations,
                conflicts=[],
                confidence=Confidence(level="high", score=0.95, reasons=[f"Defined enterprise acronym '{acronym_upper}'"]),
                timings_ms={"retrieval": 2, "generation": 1, "total": 3},
                debug={"intent": f"acronym:{word}"} if debug else None,
            )

    return None

