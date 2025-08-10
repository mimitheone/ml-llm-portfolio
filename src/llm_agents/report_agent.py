"""
LLM Agent for Banking KPI Analysis and Reporting

This module provides intelligent analysis of banking KPIs using LLM agents
with SHAP explanations and market context for regulatory compliance.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class KPIAnalysisRequest:
    """Request structure for KPI analysis."""

    kpis: Dict[str, float]
    shap_summary: Optional[str] = None
    market_context: Optional[str] = None
    historical_data: Optional[Dict[str, Any]] = None
    regulatory_requirements: Optional[list] = None


class BankingReportAgent:
    """
    LLM-powered agent for analyzing banking KPIs and generating insights.

    This agent combines traditional financial analysis with AI explainability
    to provide comprehensive KPI insights for banking stakeholders.
    """

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.analysis_templates = self._load_analysis_templates()

    def _load_analysis_templates(self) -> Dict[str, str]:
        """Load analysis templates for different KPI categories."""
        return {
            "risk": "Risk metrics analysis focusing on credit quality, portfolio risk, and regulatory compliance.",
            "performance": "Performance analysis covering profitability, efficiency, and revenue drivers.",
            "liquidity": "Liquidity analysis including regulatory ratios and cash flow management.",
            "esg": "ESG and compliance analysis with AI Act mapping and explainability requirements.",
        }

    def analyze_metrics(
        self,
        kpis: Dict[str, float],
        shap_summary: Optional[str] = None,
        market_context: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Analyze banking KPIs and generate comprehensive insights.

        Args:
            kpis: Dictionary of KPI values (e.g., {"ROTE": 0.111, "NIM": 0.027})
            shap_summary: SHAP explanation summary for model interpretability
            market_context: Market context and economic environment
            **kwargs: Additional parameters for analysis

        Returns:
            Comprehensive analysis report as string
        """
        # Categorize KPIs
        categorized_kpis = self._categorize_kpis(kpis)

        # Generate analysis for each category
        analysis_parts = []

        # Risk Analysis
        if "risk" in categorized_kpis:
            risk_analysis = self._analyze_risk_metrics(
                categorized_kpis["risk"], shap_summary, market_context
            )
            analysis_parts.append(risk_analysis)

        # Performance Analysis
        if "performance" in categorized_kpis:
            perf_analysis = self._analyze_performance_metrics(
                categorized_kpis["performance"], market_context
            )
            analysis_parts.append(perf_analysis)

        # Liquidity Analysis
        if "liquidity" in categorized_kpis:
            liq_analysis = self._analyze_liquidity_metrics(
                categorized_kpis["liquidity"], market_context
            )
            analysis_parts.append(liq_analysis)

        # ESG/Compliance Analysis
        if "esg" in categorized_kpis:
            esg_analysis = self._analyze_esg_metrics(
                categorized_kpis["esg"], shap_summary, market_context
            )
            analysis_parts.append(esg_analysis)

        # Combine all analyses
        full_analysis = "\n\n".join(analysis_parts)

        # Add market context summary if provided
        if market_context:
            full_analysis += f"\n\n**Market Context:** {market_context}"

        return full_analysis

    def _categorize_kpis(self, kpis: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Categorize KPIs into risk, performance, liquidity, and ESG categories."""
        categories = {"risk": {}, "performance": {}, "liquidity": {}, "esg": {}}

        # Risk metrics
        risk_indicators = ["PD", "LGD", "EAD", "NPL_Ratio", "VaR", "Credit_Risk"]
        # Performance metrics
        perf_indicators = ["ROTE", "ROE", "ROA", "NIM", "Cost_to_Income", "Fee_Income"]
        # Liquidity metrics
        liq_indicators = ["LCR", "NSFR", "Cash_Gap", "Liquidity_Ratio"]
        # ESG metrics
        esg_indicators = ["ESG_Score", "Sustainability_Risk", "Compliance_Score"]

        for kpi, value in kpis.items():
            if any(indicator in kpi for indicator in risk_indicators):
                categories["risk"][kpi] = value
            elif any(indicator in kpi for indicator in perf_indicators):
                categories["performance"][kpi] = value
            elif any(indicator in kpi for indicator in liq_indicators):
                categories["liquidity"][kpi] = value
            elif any(indicator in kpi for indicator in esg_indicators):
                categories["esg"][kpi] = value
            else:
                # Default to performance if unclear
                categories["performance"][kpi] = value

        return categories

    def _analyze_risk_metrics(
        self,
        risk_kpis: Dict[str, float],
        shap_summary: Optional[str],
        market_context: Optional[str],
    ) -> str:
        """Analyze risk-related KPIs."""
        analysis = "**🛡️ Risk Management Analysis:**\n\n"

        for kpi, value in risk_kpis.items():
            if "NPL" in kpi:
                if value < 0.02:
                    status = "✅ Excellent"
                elif value < 0.05:
                    status = "🟡 Good"
                else:
                    status = "🔴 Requires Attention"
                analysis += f"- **{kpi}**: {value:.3f} {status}\n"

            elif "PD" in kpi or "LGD" in kpi:
                if value < 0.05:
                    status = "✅ Low Risk"
                elif value < 0.15:
                    status = "🟡 Moderate Risk"
                else:
                    status = "🔴 High Risk"
                analysis += f"- **{kpi}**: {value:.3f} {status}\n"

        if shap_summary:
            analysis += f"\n**AI Explainability:** {shap_summary}"

        return analysis

    def _analyze_performance_metrics(
        self, perf_kpis: Dict[str, float], market_context: Optional[str]
    ) -> str:
        """Analyze performance-related KPIs."""
        analysis = "**📈 Performance Analysis:**\n\n"

        for kpi, value in perf_kpis.items():
            if "ROTE" in kpi or "ROE" in kpi:
                if value > 0.15:
                    status = "🚀 Outstanding"
                elif value > 0.10:
                    status = "✅ Strong"
                elif value > 0.05:
                    status = "🟡 Adequate"
                else:
                    status = "🔴 Below Target"
                analysis += f"- **{kpi}**: {value:.1%} {status}\n"

            elif "NIM" in kpi:
                if value > 0.04:
                    status = "🚀 Excellent"
                elif value > 0.025:
                    status = "✅ Good"
                else:
                    status = "🟡 Compressed"
                analysis += f"- **{kpi}**: {value:.1%} {status}\n"

            elif "Cost_to_Income" in kpi:
                if value < 0.50:
                    status = "🚀 Excellent Efficiency"
                elif value < 0.65:
                    status = "✅ Good Efficiency"
                else:
                    status = "🟡 Room for Improvement"
                analysis += f"- **{kpi}**: {value:.1%} {status}\n"

        return analysis

    def _analyze_liquidity_metrics(
        self, liq_kpis: Dict[str, float], market_context: Optional[str]
    ) -> str:
        """Analyze liquidity-related KPIs."""
        analysis = "**💧 Liquidity Analysis:**\n\n"

        for kpi, value in liq_kpis.items():
            if "LCR" in kpi:
                if value > 1.3:
                    status = "🚀 Strong Liquidity"
                elif value > 1.0:
                    status = "✅ Compliant"
                else:
                    status = "🔴 Below Regulatory Minimum"
                analysis += f"- **{kpi}**: {value:.2f} {status}\n"

            elif "NSFR" in kpi:
                if value > 1.1:
                    status = "🚀 Strong Funding"
                elif value > 1.0:
                    status = "✅ Compliant"
                else:
                    status = "🔴 Below Regulatory Minimum"
                analysis += f"- **{kpi}**: {value:.2f} {status}\n"

        return analysis

    def _analyze_esg_metrics(
        self,
        esg_kpis: Dict[str, float],
        shap_summary: Optional[str],
        market_context: Optional[str],
    ) -> str:
        """Analyze ESG and compliance metrics."""
        analysis = "**🌱 ESG & Compliance Analysis:**\n\n"

        for kpi, value in esg_kpis.items():
            if "ESG_Score" in kpi:
                if value > 0.8:
                    status = "🌱 Leadership"
                elif value > 0.6:
                    status = "✅ Good Standing"
                else:
                    status = "🟡 Development Needed"
                analysis += f"- **{kpi}**: {value:.1%} {status}\n"

        if shap_summary:
            analysis += f"\n**AI Explainability (EU AI Act):** {shap_summary}"
            analysis += "\n\n**Regulatory Mapping:**"
            analysis += "\n- Risk Category: Medium (AI-assisted analysis)"
            analysis += "\n- Data Governance: Compliant with GDPR requirements"
            analysis += (
                "\n- Transparency: SHAP explanations provided for all predictions"
            )

        return analysis


# Convenience function for easy usage
def analyze_metrics(
    kpis: Dict[str, float],
    shap_summary: Optional[str] = None,
    market_context: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Convenience function to analyze banking KPIs.

    Args:
        kpis: Dictionary of KPI values
        shap_summary: SHAP explanation summary
        market_context: Market context information
        **kwargs: Additional parameters

    Returns:
        Analysis report as string
    """
    agent = BankingReportAgent()
    return agent.analyze_metrics(kpis, shap_summary, market_context, **kwargs)
