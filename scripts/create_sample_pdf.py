#!/usr/bin/env python3
"""
Create a sample technical PDF for testing the RAG system.
Generates a realistic enterprise API documentation with multiple pages,
headings, sections, and technical content.
"""

import pymupdf  # PyMuPDF


def create_sample_pdf(output_path: str = "data/uploads/sample_api_docs.pdf"):
    """Create a sample API documentation PDF."""
    doc = pymupdf.open()

    PAGE_WIDTH = 612
    PAGE_HEIGHT = 792
    MARGIN = 50
    CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

    def add_page():
        return doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    def write_text(page, x, y, text, fontsize=11, fontname="helv", color=(0, 0, 0)):
        page.insert_text((x, y), text, fontsize=fontsize, fontname=fontname, color=color)
        # Estimate height for positioning
        return fontsize * 1.5

    def write_box(page, x, y, width, height, text, fontsize=11, fontname="helv"):
        rect = pymupdf.Rect(x, y, x + width, y + height)
        page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname)
        return height

    # Page 1 - Title & Overview
    page = add_page()
    y = 50

    y += write_text(page, MARGIN, y, "Enterprise API Authentication Guide", fontsize=24, fontname="helv")
    y += 20
    y += write_text(page, MARGIN, y, "Version 2.1 | Updated: 2024-01-15", fontsize=14, fontname="helv", color=(0.4, 0.4, 0.4))
    y += 20

    y += write_text(page, MARGIN, y, "1. Overview", fontsize=18, fontname="helv")
    y += 15

    overview_text = (
        "This document describes the authentication mechanisms used by the Enterprise API platform. "
        "All API endpoints require authentication except for the health check endpoint. "
        "The platform supports two primary authentication methods: JWT (JSON Web Tokens) and API Keys. "
        "OAuth 2.0 is supported for third-party integrations."
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 100, overview_text, fontsize=11) + 10

    y += write_text(page, MARGIN, y, "1.1 Architecture", fontsize=14, fontname="helv")
    y += 10

    arch_text = (
        "The authentication system follows a microservices architecture. The Auth Service handles token "
        "issuance, validation, and revocation. The API Gateway performs initial token validation and "
        "routes requests to appropriate backend services. All services share a common Redis cache for "
        "token blacklisting and rate limiting."
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 80, arch_text, fontsize=11) + 10

    # Page 2 - JWT Authentication
    page = add_page()
    y = 50

    y += write_text(page, MARGIN, y, "2. JWT Authentication", fontsize=18, fontname="helv")
    y += 15

    jwt_intro = (
        "JWT (JSON Web Token) is the primary authentication mechanism for the Enterprise API. "
        "Tokens are issued by the Auth Service after successful user authentication. "
        "All JWTs use RS256 (RSA Signature with SHA-256) algorithm for signing."
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 60, jwt_intro, fontsize=11) + 10

    y += write_text(page, MARGIN, y, "2.1 Token Structure", fontsize=14, fontname="helv")
    y += 10

    structure_text = (
        "A JWT consists of three parts separated by dots: header.payload.signature\n\n"
        "Header:\n"
        "  {\n"
        "    \"alg\": \"RS256\",\n"
        "    \"typ\": \"JWT\",\n"
        "    \"kid\": \"key-id-123\"\n"
        "  }\n\n"
        "Payload (Claims):\n"
        "  {\n"
        "    \"sub\": \"user-12345\",\n"
        "    \"iss\": \"https://auth.enterprise.com\",\n"
        "    \"aud\": \"api.enterprise.com\",\n"
        "    \"exp\": 1705312800,\n"
        "    \"iat\": 1705226400,\n"
        "    \"scope\": \"read write admin\",\n"
        "    \"tenant_id\": \"tenant-abc\"\n"
        "  }"
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 200, structure_text, fontsize=10, fontname="courier") + 10

    y += write_text(page, MARGIN, y, "2.2 Token Usage", fontsize=14, fontname="helv")
    y += 10

    usage_text = (
        "Include the JWT in the Authorization header of every API request:\n\n"
        "  Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...\n\n"
        "Tokens expire after 24 hours (configurable per tenant). Refresh tokens can be used to "
        "obtain new access tokens without re-authentication. The refresh token endpoint is "
        "POST /auth/refresh with the refresh token in the request body."
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 120, usage_text, fontsize=11) + 10

    # Page 3 - API Key Authentication
    page = add_page()
    y = 50

    y += write_text(page, MARGIN, y, "3. API Key Authentication", fontsize=18, fontname="helv")
    y += 15

    api_key_intro = (
        "API Keys provide a simpler authentication method for service-to-service communication "
        "and long-running integrations. Unlike JWTs, API Keys do not expire automatically but "
        "can be revoked at any time."
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 60, api_key_intro, fontsize=11) + 10

    y += write_text(page, MARGIN, y, "3.1 Key Format", fontsize=14, fontname="helv")
    y += 10

    format_text = (
        "Enterprise API Keys follow the format: ent_live_<random_string>\n\n"
        "Example: ent_live_sk_abc123def456ghi789jkl\n\n"
        "Keys are prefixed with 'ent_live_' for production and 'ent_test_' for sandbox environments. "
        "The prefix allows easy identification of key environments in logs and monitoring."
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 100, format_text, fontsize=11) + 10

    y += write_text(page, MARGIN, y, "3.2 Key Usage", fontsize=14, fontname="helv")
    y += 10

    key_usage_text = (
        "Include the API Key in the X-API-Key header:\n\n"
        "  X-API-Key: ent_live_sk_abc123def456ghi789jkl\n\n"
        "API Keys inherit the permissions of the service account that created them. "
        "Rotate keys regularly using the Key Management API (POST /keys/rotate). "
        "A maximum of 5 active keys per service account is enforced."
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 100, key_usage_text, fontsize=11) + 10

    y += write_text(page, MARGIN, y, "3.3 Key Security", fontsize=14, fontname="helv")
    y += 10

    security_text = (
        "• Never commit API Keys to version control\n"
        "• Store keys in secure secret management systems (HashiCorp Vault, AWS Secrets Manager)\n"
        "• Use environment variables for application configuration\n"
        "• Monitor key usage via the Audit Log API\n"
        "• Enable IP allowlisting for production keys where possible"
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 80, security_text, fontsize=11) + 10

    # Page 4 - OAuth 2.0 Integration
    page = add_page()
    y = 50

    y += write_text(page, MARGIN, y, "4. OAuth 2.0 Integration", fontsize=18, fontname="helv")
    y += 15

    oauth_text = (
        "Third-party applications can integrate using OAuth 2.0 Authorization Code flow with PKCE. "
        "The Enterprise API acts as an OAuth 2.0 Resource Server and Authorization Server."
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 60, oauth_text, fontsize=11) + 10

    y += write_text(page, MARGIN, y, "4.1 Endpoints", fontsize=14, fontname="helv")
    y += 10

    endpoints_text = (
        "Authorization Endpoint:  GET  https://auth.enterprise.com/oauth/authorize\n"
        "Token Endpoint:          POST https://auth.enterprise.com/oauth/token\n"
        "Revocation Endpoint:     POST https://auth.enterprise.com/oauth/revoke\n"
        "Introspection Endpoint:  POST https://auth.enterprise.com/oauth/introspect\n"
        "JWKS Endpoint:           GET  https://auth.enterprise.com/.well-known/jwks.json"
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 100, endpoints_text, fontsize=10, fontname="courier") + 10

    y += write_text(page, MARGIN, y, "4.2 Scopes", fontsize=14, fontname="helv")
    y += 10

    scopes_text = (
        "Available OAuth scopes:\n\n"
        "• read:profile     - Read user profile information\n"
        "• read:data        - Read access to API resources\n"
        "• write:data       - Write access to API resources\n"
        "• admin:tenants    - Tenant administration (requires approval)\n"
        "• admin:keys       - API Key management\n\n"
        "Request only the scopes your application needs. Excessive scope requests "
        "will be rejected during the consent screen."
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 120, scopes_text, fontsize=11) + 10

    # Page 5 - Error Handling & Troubleshooting
    page = add_page()
    y = 50

    y += write_text(page, MARGIN, y, "5. Error Handling & Troubleshooting", fontsize=18, fontname="helv")
    y += 15

    y += write_text(page, MARGIN, y, "5.1 Common Error Codes", fontsize=14, fontname="helv")
    y += 10

    errors_text = (
        "401 Unauthorized - Invalid or missing authentication\n"
        "  • JWT: Token expired, invalid signature, or malformed token\n"
        "  • API Key: Key revoked, invalid format, or wrong environment\n"
        "  • OAuth: Invalid token, insufficient scope, or token revoked\n\n"
        "403 Forbidden - Valid authentication but insufficient permissions\n"
        "  • Required scope not granted\n"
        "  • Tenant access denied\n"
        "  • Resource ownership mismatch\n\n"
        "429 Too Many Requests - Rate limit exceeded\n"
        "  • Default: 1000 requests/minute per API Key\n"
        "  • Default: 5000 requests/minute per JWT\n"
        "  • Response includes Retry-After header\n\n"
        "503 Service Unavailable - Auth Service temporarily unavailable\n"
        "  • Implement exponential backoff with jitter\n"
        "  • Cache valid tokens locally for resilience"
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 200, errors_text, fontsize=10) + 10

    y += write_text(page, MARGIN, y, "5.2 Debugging Tips", fontsize=14, fontname="helv")
    y += 10

    debug_text = (
        "1. Check the X-Request-ID header in responses for tracing\n"
        "2. Use the Introspection endpoint to validate tokens\n"
        "3. Verify JWKS endpoint for public key rotation\n"
        "4. Check Audit Logs for authentication events\n"
        "5. Test with the Sandbox environment first (ent_test_ keys)"
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 80, debug_text, fontsize=11) + 10

    # Page 6 - Rate Limiting & Best Practices
    page = add_page()
    y = 50

    y += write_text(page, MARGIN, y, "6. Rate Limiting & Best Practices", fontsize=18, fontname="helv")
    y += 15

    y += write_text(page, MARGIN, y, "6.1 Rate Limits", fontsize=14, fontname="helv")
    y += 10

    limits_text = (
        "Rate limits are applied per authentication method:\n\n"
        "JWT Authentication:\n"
        "  • 5,000 requests/minute per token\n"
        "  • 50,000 requests/hour per token\n"
        "  • Burst allowance: 100 requests/second\n\n"
        "API Key Authentication:\n"
        "  • 1,000 requests/minute per key\n"
        "  • 10,000 requests/hour per key\n"
        "  • Burst allowance: 50 requests/second\n\n"
        "OAuth 2.0:\n"
        "  • 2,000 requests/minute per access token\n"
        "  • 20,000 requests/hour per access token\n\n"
        "Rate limit headers in responses:\n"
        "  • X-RateLimit-Limit: Request limit in current window\n"
        "  • X-RateLimit-Remaining: Requests remaining\n"
        "  • X-RateLimit-Reset: Unix timestamp when limit resets"
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 220, limits_text, fontsize=10) + 10

    y += write_text(page, MARGIN, y, "6.2 Best Practices", fontsize=14, fontname="helv")
    y += 10

    best_practices = (
        "• Always handle 429 responses with exponential backoff\n"
        "• Cache valid tokens to reduce Auth Service calls\n"
        "• Use short-lived JWTs (15-30 min) for high-security contexts\n"
        "• Implement token refresh before expiration (5 min buffer)\n"
        "• Monitor authentication metrics via /metrics endpoint\n"
        "• Rotate API Keys quarterly\n"
        "• Use mTLS for service-to-service communication where possible"
    )
    y += write_box(page, MARGIN, y, CONTENT_WIDTH, 100, best_practices, fontsize=11) + 10

    # Save
    doc.save(output_path)
    doc.close()
    print(f"Sample PDF created at: {output_path}")
    print(f"Pages: 6")


if __name__ == "__main__":
    create_sample_pdf()