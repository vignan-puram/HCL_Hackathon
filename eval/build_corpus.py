"""Render the two PDF documents of the synthetic corpus with PyMuPDF.

The PDFs use real font-size headings, span several pages and carry a repeated
header/footer, so the PDF parser's heading detection and header stripping are
exercised. Run from repo root: python -m eval.build_corpus
"""

from __future__ import annotations

import textwrap

import pymupdf as fitz

import config

PAGE_W, PAGE_H, MARGIN = 595, 842, 60
SIZES = {"title": 20, "h1": 15, "h2": 12.5, "body": 10.5, "meta": 10.5}
NEW_PAGE = ("newpage", "")

ADMIN_GUIDE = [
    ("title", "Nimbus Workspace Administrator Guide"),
    ("meta", "Version 3.0 | Effective date: 2025-02-15"),
    ("body", "This guide is for workspace Owners and Admins. It covers roles, single sign-on, "
             "password resets, API keys and audit logs in Nimbus Workspace 3.0."),
    ("h1", "1. Roles and Permissions"),
    ("body", "Every member of a workspace has exactly one role. Owners manage billing, can delete the "
             "workspace and can transfer ownership. Admins manage users, roles, security settings and "
             "API keys. Members create and share content. Guests can only open items that are shared "
             "with them and cannot invite other people."),
    ("body", "Only Owners can create new workspaces through the API. Role changes take effect "
             "immediately and are recorded in the audit log."),
    NEW_PAGE,
    ("h1", "2. Single Sign-On (SSO)"),
    ("h2", "2.1 Configuring SAML"),
    ("body", "Nimbus Workspace supports single sign-on with any SAML 2.0 identity provider. To configure "
             "SAML, open Admin Console > Security > Single sign-on, select Add identity provider, and "
             "upload the metadata XML file from your identity provider. Copy the Nimbus Assertion "
             "Consumer Service URL and Entity ID into your identity provider, then select Test "
             "connection. Enable Enforce SSO only after a successful test, otherwise admins can be "
             "locked out."),
    ("h2", "2.2 Just-in-time provisioning"),
    ("body", "When just-in-time provisioning is enabled, users who sign in through the identity provider "
             "for the first time are created automatically with the Member role. Admins can change the "
             "default role in the SSO settings."),
    NEW_PAGE,
    ("h1", "3. Resetting a User's Password"),
    ("body", "Admins can reset the password of any member who does not sign in with SSO. Open Admin "
             "Console > Users, select the user, and choose Reset password. Nimbus sends the user an "
             "email with a reset link. The reset link expires after 24 hours. Users who sign in with "
             "SSO must reset their password in the identity provider instead."),
    ("body", "After 5 failed sign-in attempts an account is locked for 15 minutes. An admin can unlock "
             "the account immediately from the same Users page."),
    ("h1", "4. API Keys"),
    ("body", "Admins create API keys in Admin Console > Developer > API keys. Each workspace can have up "
             "to 10 active API keys. Give each integration its own key so that it can be revoked "
             "without affecting other integrations. Nimbus recommends rotating API keys every 90 days. "
             "A revoked key stops working immediately and requests using it fail with ERR-401."),
    NEW_PAGE,
    ("h1", "5. Audit Logs"),
    ("body", "The audit log records sign-ins, role changes, API key events and file sharing. Admins can "
             "filter events by user, date and event type in Admin Console > Security > Audit log, and "
             "export them as CSV."),
]

BILLING_POLICY = [
    ("title", "Billing Policy"),
    ("meta", "Version 2.0 | Effective date: 2025-03-01"),
    ("body", "This policy sets out how Nimbus Workspace subscriptions are priced, invoiced and refunded. "
             "It replaces all earlier billing documents."),
    ("h1", "1. Plans and Pricing"),
    ("body", "Nimbus Workspace is offered on three plans. The Free plan costs nothing. The Pro plan costs "
             "$12 per user per month when billed monthly, or $10 per user per month when billed annually. "
             "Enterprise pricing is agreed in a contract with the sales team."),
    ("h1", "2. Billing Cycles"),
    ("body", "Subscriptions are billed monthly or annually in advance. Adding users during a billing "
             "period creates a prorated charge on the next invoice. Removing users takes effect at the "
             "next renewal."),
    NEW_PAGE,
    ("h1", "3. Refunds"),
    ("h2", "3.1 Monthly plans"),
    ("body", "Monthly subscriptions are not refundable. When a monthly subscription is cancelled, the "
             "workspace remains on the paid plan until the end of the current month."),
    ("h2", "3.2 Annual plans"),
    ("body", "Annual subscriptions may be refunded within 30 days of purchase. Refunds are issued to the "
             "original payment method. After 30 days, annual subscriptions can be cancelled but the "
             "remaining months are not refunded."),
    ("h1", "4. Failed Payments"),
    ("body", "If a payment fails, Nimbus retries the charge and notifies the workspace Owner. There is a "
             "grace period of 7 days. If payment is still outstanding after the grace period, the "
             "workspace is downgraded to the Free plan; no data is deleted."),
    ("h1", "5. Taxes"),
    ("body", "Prices exclude sales tax and VAT. Applicable taxes are added to each invoice based on the "
             "billing address."),
]


def _new_page(doc: fitz.Document, header: str) -> tuple[fitz.Page, float]:
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((MARGIN, 35), header, fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))
    return page, MARGIN + 20


def render(blocks: list[tuple[str, str]], header: str, out_name: str) -> int:
    doc = fitz.open()
    page, y = _new_page(doc, header)
    for kind, text in blocks:
        if kind == "newpage":
            page, y = _new_page(doc, header)
            continue
        size = SIZES[kind]
        font = "hebo" if kind in ("title", "h1", "h2") else "helv"
        width_chars = int((PAGE_W - 2 * MARGIN) / (size * 0.5))
        lines = textwrap.wrap(text, width_chars)
        needed = len(lines) * size * 1.45 + size
        if y + needed > PAGE_H - MARGIN:
            page, y = _new_page(doc, header)
        if kind in ("h1", "h2"):
            y += size * 0.6
        for line in lines:
            y += size * 1.45
            page.insert_text((MARGIN, y), line, fontsize=size, fontname=font)
        y += size * 0.6
    for number, pg in enumerate(doc, start=1):
        pg.insert_text((MARGIN, PAGE_H - 30), f"Nimbus Labs (fictional) - Page {number}",
                       fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))
    doc.set_metadata({"title": blocks[0][1]})
    out = config.CORPUS_DIR / out_name
    doc.save(out)
    return doc.page_count


def main() -> None:
    pages = render(ADMIN_GUIDE, "Nimbus Workspace - Administrator Guide", "admin_guide_v3.pdf")
    print(f"admin_guide_v3.pdf: {pages} pages")
    pages = render(BILLING_POLICY, "Nimbus Workspace - Billing Policy", "billing_policy_v2.pdf")
    print(f"billing_policy_v2.pdf: {pages} pages")


if __name__ == "__main__":
    main()
