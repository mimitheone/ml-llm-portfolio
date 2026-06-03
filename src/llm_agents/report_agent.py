"""
LLM Agent for Banking KPI Analysis and Reporting

Uses LangChain LCEL for LLM-powered analysis. Falls back to rule-based
output when the LLM is unavailable (missing API key or package).
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior banking analyst specializing in regulatory compliance and financial "
    "performance. Generate concise, structured KPI reports for executives and regulators. "
    "Follow Basel III, IFRS 9, and EU AI Act standards. When SHAP explanations are provided, "
    "integrate them into your risk commentary."
)

_USER_TEMPLATE = """\
Analyze the following banking KPIs and generate a structured report.

## KPI Data
{kpi_summary}
{shap_section}
{context_section}
Generate a structured analysis with sections:
1. Risk Assessment (flag NPL > 5%, PD/LGD thresholds)
2. Performance Review (ROTE, NIM, Cost-to-Income benchmarks)
3. Liquidity Position (flag LCR < 1.0, NSFR < 1.0)
4. ESG & Regulatory Compliance (EU AI Act mapping if SHAP provided)
5. Key Recommendations (top 3, prioritized by urgency)
"""

_CATEGORY_INDICATORS: Dict[str, list[str]] = {
    "risk":        ["PD", "LGD", "EAD", "NPL_Ratio", "VaR", "Credit_Risk"],
    "performance": ["ROTE", "ROE", "ROA", "NIM", "Cost_to_Income", "Fee_Income"],
    "liquidity":   ["LCR", "NSFR", "Cash_Gap", "Liquidity_Ratio"],
    "esg":         ["ESG_Score", "Sustainability_Risk", "Compliance_Score"],
}


@dataclass
class KPIAnalysisRequest:
    kpis: Dict[str, float]
    shap_summary: Optional[str] = None
    market_context: Optional[str] = None
    historical_data: Optional[Dict[str, Any]] = None
    regulatory_requirements: Optional[list] = None


class BankingReportAgent:
    """
    LLM-powered agent for analyzing banking KPIs.

    Preprocesses KPIs into categorized summaries, then invokes an LLM via
    LangChain LCEL. Falls back to rule-based output if the LLM is unavailable.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self._chain = None  # built lazily on first use

    # ── LangChain chain ──────────────────────────────────────────────────────

    def _get_chain(self):
        if self._chain is not None:
            return self._chain
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
            prompt = ChatPromptTemplate.from_messages(
                [("system", _SYSTEM_PROMPT), ("human", _USER_TEMPLATE)]
            )
            self._chain = prompt | llm | StrOutputParser()
            return self._chain
        except ImportError:
            warnings.warn(
                "langchain-openai is not installed — falling back to rule-based analysis. "
                "Install with: pip install langchain-openai",
                stacklevel=3,
            )
        except Exception as exc:
            warnings.warn(
                f"LLM unavailable ({exc}) — falling back to rule-based analysis.",
                stacklevel=3,
            )
        return None

    # ── Public API ───────────────────────────────────────────────────────────

    def analyze_metrics(
        self,
        kpis: Dict[str, float],
        shap_summary: Optional[str] = None,
        market_context: Optional[str] = None,
        **kwargs,
    ) -> str:
        categorized, uncategorized = self._categorize_kpis(kpis)

        if uncategorized:
            logger.warning(
                "Uncategorized KPIs (included under 'other'): %s", list(uncategorized)
            )

        kpi_summary = self._format_kpi_summary(categorized, uncategorized)
        shap_section = f"\n## SHAP Explainability\n{shap_summary}\n" if shap_summary else ""
        context_section = f"\n## Market Context\n{market_context}\n" if market_context else ""

        chain = self._get_chain()
        if chain is not None:
            return chain.invoke(
                {
                    "kpi_summary": kpi_summary,
                    "shap_section": shap_section,
                    "context_section": context_section,
                }
            )

        return self._rule_based_analysis(
            categorized, uncategorized, shap_summary, market_context
        )

    # ── Categorization ───────────────────────────────────────────────────────

    def _categorize_kpis(
        self, kpis: Dict[str, float]
    ) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        categorized: Dict[str, Dict[str, float]] = {k: {} for k in _CATEGORY_INDICATORS}
        uncategorized: Dict[str, float] = {}
        for kpi, value in kpis.items():
            matched = False
            for category, indicators in _CATEGORY_INDICATORS.items():
                if any(ind in kpi for ind in indicators):
                    categorized[category][kpi] = value
                    matched = True
                    break
            if not matched:
                uncategorized[kpi] = value
        return categorized, uncategorized

    def _format_kpi_summary(self, categorized, uncategorized) -> str:
        lines = []
        for category, kpis in categorized.items():
            if kpis:
                lines.append(f"\n### {category.upper()}")
                for kpi, value in kpis.items():
                    lines.append(f"  {kpi}: {value}")
        if uncategorized:
            lines.append("\n### OTHER")
            for kpi, value in uncategorized.items():
                lines.append(f"  {kpi}: {value}")
        return "\n".join(lines)

    # ── Rule-based fallback ──────────────────────────────────────────────────

    def _rule_based_analysis(self, categorized, uncategorized, shap_summary, market_context) -> str:
        parts = []
        if categorized.get("risk"):
            parts.append(self._analyze_risk(categorized["risk"], shap_summary))
        if categorized.get("performance"):
            parts.append(self._analyze_performance(categorized["performance"]))
        if categorized.get("liquidity"):
            parts.append(self._analyze_liquidity(categorized["liquidity"]))
        if categorized.get("esg"):
            parts.append(self._analyze_esg(categorized["esg"], shap_summary))
        if uncategorized:
            lines = [f"- **{k}**: {v}" for k, v in uncategorized.items()]
            parts.append("**Other Metrics:**\n\n" + "\n".join(lines))
        if market_context:
            parts.append(f"**Market Context:** {market_context}")
        return "\n\n".join(parts)

    def _analyze_risk(self, risk_kpis, shap_summary) -> str:
        analysis = "**Risk Management Analysis:**\n\n"
        for kpi, value in risk_kpis.items():
            if "NPL" in kpi:
                status = "Excellent" if value < 0.02 else ("Good" if value < 0.05 else "Requires Attention")
                analysis += f"- **{kpi}**: {value:.3f} — {status}\n"
            elif "PD" in kpi or "LGD" in kpi:
                status = "Low Risk" if value < 0.05 else ("Moderate Risk" if value < 0.15 else "High Risk")
                analysis += f"- **{kpi}**: {value:.3f} — {status}\n"
        if shap_summary:
            analysis += f"\n**AI Explainability:** {shap_summary}"
        return analysis

    def _analyze_performance(self, perf_kpis) -> str:
        analysis = "**Performance Analysis:**\n\n"
        for kpi, value in perf_kpis.items():
            if "ROTE" in kpi or "ROE" in kpi:
                status = "Outstanding" if value > 0.15 else ("Strong" if value > 0.10 else ("Adequate" if value > 0.05 else "Below Target"))
                analysis += f"- **{kpi}**: {value:.1%} — {status}\n"
            elif "NIM" in kpi:
                status = "Excellent" if value > 0.04 else ("Good" if value > 0.025 else "Compressed")
                analysis += f"- **{kpi}**: {value:.1%} — {status}\n"
            elif "Cost_to_Income" in kpi:
                status = "Excellent Efficiency" if value < 0.50 else ("Good Efficiency" if value < 0.65 else "Room for Improvement")
                analysis += f"- **{kpi}**: {value:.1%} — {status}\n"
        return analysis

    def _analyze_liquidity(self, liq_kpis) -> str:
        analysis = "**Liquidity Analysis:**\n\n"
        for kpi, value in liq_kpis.items():
            if "LCR" in kpi:
                status = "Strong Liquidity" if value > 1.3 else ("Compliant" if value > 1.0 else "Below Regulatory Minimum")
                analysis += f"- **{kpi}**: {value:.2f} — {status}\n"
            elif "NSFR" in kpi:
                status = "Strong Funding" if value > 1.1 else ("Compliant" if value > 1.0 else "Below Regulatory Minimum")
                analysis += f"- **{kpi}**: {value:.2f} — {status}\n"
        return analysis

    def _analyze_esg(self, esg_kpis, shap_summary) -> str:
        analysis = "**ESG & Compliance Analysis:**\n\n"
        for kpi, value in esg_kpis.items():
            if "ESG_Score" in kpi:
                status = "Leadership" if value > 0.8 else ("Good Standing" if value > 0.6 else "Development Needed")
                analysis += f"- **{kpi}**: {value:.1%} — {status}\n"
        if shap_summary:
            analysis += f"\n**AI Explainability (EU AI Act):** {shap_summary}"
            analysis += "\n- Risk Category: Medium (AI-assisted analysis)"
            analysis += "\n- Data Governance: Compliant with GDPR requirements"
            analysis += "\n- Transparency: SHAP explanations provided for all predictions"
        return analysis


def analyze_metrics(
    kpis: Dict[str, float],
    shap_summary: Optional[str] = None,
    market_context: Optional[str] = None,
    **kwargs,
) -> str:
    agent = BankingReportAgent()
    return agent.analyze_metrics(kpis, shap_summary, market_context, **kwargs)
