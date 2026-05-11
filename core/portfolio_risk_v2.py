"""
Portfolio Risk v2 - Exposure Control Layer

Controls:
- Currency exposure limits
- Correlation cluster control
- Net portfolio exposure cap
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PortfolioRiskSettings:
    """Settings for portfolio risk control."""
    enabled: bool = False
    
    # Per currency limits
    max_currency_exposure: int = 2
    max_currency_gross_exposure: int = 4
    max_currency_net_exposure: int = 2
    
    # Correlation cluster
    correlation_threshold: float = 0.82
    max_cluster_exposure: int = 3
    
    # Net portfolio
    max_net_direction: int = 4
    
    def sanitized(self) -> "PortfolioRiskSettings":
        return PortfolioRiskSettings(
            enabled=self.enabled,
            max_currency_exposure=max(0, int(self.max_currency_exposure)),
            max_currency_gross_exposure=max(0, int(self.max_currency_gross_exposure)),
            max_currency_net_exposure=max(0, int(self.max_currency_net_exposure)),
            correlation_threshold=max(0.0, float(self.correlation_threshold)),
            max_cluster_exposure=max(0, int(self.max_cluster_exposure)),
            max_net_direction=max(0, int(self.max_net_direction)),
        )


@dataclass
class CurrencyExposure:
    """Track currency exposure."""
    currency: str
    long_count: int = 0
    short_count: int = 0
    
    @property
    def net(self) -> int:
        return self.long_count - self.short_count
    
    @property
    def gross(self) -> int:
        return self.long_count + self.short_count
    
    def add_long(self) -> None:
        self.long_count += 1
    
    def add_short(self) -> None:
        self.short_count += 1


class PortfolioRiskV2:
    """Portfolio risk controller."""
    
    def __init__(
        self,
        settings: PortfolioRiskSettings | None = None,
        correlation_data: dict[str, float] | None = None,
    ):
        self.settings = (settings or PortfolioRiskSettings()).sanitized()
        self.correlation_data = correlation_data or {}
        
        # Active exposures
        self._exposures: dict[str, CurrencyExposure] = {}
        self._correlation_clusters: dict[str, list[str]] = {}
    
    def _get_pair_currencies(self, pair: str) -> tuple[str, str]:
        """Extract currencies from pair."""
        clean = pair.upper().replace("/", "")
        if len(clean) != 6:
            return "", ""
        return clean[:3], clean[3:6]
    
    def _find_correlation_cluster(
        self,
        pair: str,
        universe: set[str],
    ) -> list[str]:
        """Find all correlated pairs in cluster."""
        if not self.correlation_data:
            return [pair]
        
        base, quote = self._get_pair_currencies(pair)
        if not base or not quote:
            return [pair]
        
        cluster = [pair]
        for other in universe:
            if other == pair:
                continue
            
            other_base, other_quote = self._get_pair_currencies(other)
            if not other_base or not other_quote:
                continue
            
            # Check correlation
            corr_key = f"{base}{other_base}"
            corr = self.correlation_data.get(corr_key, 0.0)
            if abs(corr) >= self.settings.correlation_threshold:
                cluster.append(other)
            
            # Same quote currency
            if other_quote == quote:
                corr_key2 = f"{base}{other_quote}"
                corr2 = self.correlation_data.get(corr_key2, 0.0)
                if abs(corr2) >= self.settings.correlation_threshold:
                    cluster.append(other)
        
        return cluster
    
    def _update_exposure(
        self,
        pair: str,
        side: str,  # "BUY" or "SELL"
    ) -> None:
        """Update exposure for a trade."""
        base, quote = self._get_pair_currencies(pair)
        
        if side.upper() == "BUY":
            if base not in self._exposures:
                self._exposures[base] = CurrencyExposure(base)
            self._exposures[base].add_long()
            
            if quote not in self._exposures:
                self._exposures[quote] = CurrencyExposure(quote)
            self._exposures[quote].add_short()
        else:
            if base not in self._exposures:
                self._exposures[base] = CurrencyExposure(base)
            self._exposures[base].add_short()
            
            if quote not in self._exposures:
                self._exposures[quote] = CurrencyExposure(quote)
            self._exposures[quote].add_long()
    
    def check_trade(
        self,
        pair: str,
        side: str,
        universe: set[str] | None = None,
    ) -> tuple[bool, str]:
        """
        Check if trade allowed.
        
        Returns: (allowed, reason)
        """
        if not self.settings.enabled:
            return True, ""
        
        universe = universe or set()
        base, quote = self._get_pair_currencies(pair)
        
        if not base or not quote:
            return False, "invalid_pair"
        
        # Check currency net exposure
        for currency in [base, quote]:
            exp = self._exposures.get(currency)
            if exp:
                if side.upper() == "BUY":
                    new_net = exp.net + 1
                else:
                    new_net = exp.net - 1
                
                if abs(new_net) > self.settings.max_currency_net_exposure:
                    return False, f"net_exposure_{currency}"
        
        # Check currency gross exposure
        for currency in [base, quote]:
            exp = self._exposures.get(currency)
            if exp:
                new_gross = exp.gross + 1
                if new_gross > self.settings.max_currency_gross_exposure:
                    return False, f"gross_exposure_{currency}"
        
        # Check correlation cluster
        cluster = self._find_correlation_cluster(pair, universe)
        cluster_exposure = sum(
            self._exposures.get(c, CurrencyExposure(c)).gross
            for c in cluster if c in self._exposures
        )
        if cluster_exposure >= self.settings.max_cluster_exposure:
            return False, f"cluster_exposure_{len(cluster)}"
        
        return True, ""
    
    def register_trade(
        self,
        pair: str,
        side: str,
    ) -> None:
        """Register executed trade."""
        self._update_exposure(pair, side)
    
    def get_exposure_summary(self) -> dict[str, Any]:
        """Get current exposure summary."""
        return {
            "enabled": self.settings.enabled,
            "currencies": {
                currency: {
                    "long": exp.long_count,
                    "short": exp.short_count,
                    "net": exp.net,
                    "gross": exp.gross,
                }
                for currency, exp in self._exposures.items()
            },
        }
    
    def can_trade_more(
        self,
        new_side: str,
        universe: set[str] | None = None,
    ) -> bool:
        """Check if can take more trades."""
        if not self.settings.enabled:
            return True
        
        universe = universe or set()
        
        # Count net direction
        net_buys = sum(
            exp.long_count for exp in self._exposures.values()
        )
        net_sells = sum(
            exp.short_count for exp in self._exposures.values()
        )
        
        if new_side.upper() == "BUY":
            net_buys += 1
        else:
            net_sells += 1
        
        # Check net direction limit
        if abs(net_buys - net_sells) > self.settings.max_net_direction:
            return False
        
        return True
    
    def reset(self) -> None:
        """Reset all exposures."""
        self._exposures.clear()