"""Command-line interface for the name screening tool."""

import json
import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from .core.entity_extractor import EntityExtractor
from .core.matcher import NameMatcher
from .models.matching import MatchResult

app = typer.Typer(
    name="name_screening",
    help="Match names in news articles to individuals of interest",
    add_completion=False,
)

console = Console()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_spacy_model(language: str, nicknames_path: Optional[str] = None) -> bool:
    extractor = EntityExtractor()
    if not extractor._load_model(language):
        console.print(
            f"[red]No spaCy model found for language '{language}'[/red]\n"
            f"Please install a model with: python -m spacy download en_core_web_sm"
        )
        return False
    return True


@app.command()
def match(
    name: str = typer.Option(..., "--name", "-n", help="Name to search for"),
    article_file: Optional[Path] = typer.Option(
        None, "--article-file", "-f", help="Path to article text file"
    ),
    article_text: Optional[str] = typer.Option(
        None, "--article-text", "-t", help="Article text directly"
    ),
    language: str = typer.Option(
        "en", "--lang", "-l", help="Two-letter language code (e.g., en, es)"
    ),
    strict: bool = typer.Option(
        False, "--strict", "-s", help="Use strict matching (higher confidence required)"
    ),
    use_llm: bool = typer.Option(
        False,
        "--use-llm",
        help="Enhance matching with LLM contextual analysis (requires GOOGLE_API_KEY)",
    ),
    allow_first_name_only: bool = typer.Option(
        True,
        "--allow-first-name-only/--no-first-name-only",
        help="Allow matches where only first names match (disable for high-security environments)",
    ),
    output_format: str = typer.Option(
        "pretty", "--output", "-o", help="Output format: pretty or json"
    ),
):
    if not article_file and not article_text:
        console.print(
            "[red]Error: Provide either --article-file or --article-text[/red]"
        )
        raise typer.Exit(1)

    if article_file and article_text:
        console.print(
            "[red]Error: Provide only one of --article-file or --article-text[/red]"
        )
        raise typer.Exit(1)

    if article_file:
        if not article_file.exists():
            console.print(f"[red]Error: File not found: {article_file}[/red]")
            raise typer.Exit(1)

        try:
            article_text = article_file.read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            raise typer.Exit(1)

    if not setup_spacy_model(language):
        raise typer.Exit(1)

    try:
        results = process_match_request(
            name=name,
            article_text=article_text,
            language=language,
            strict_mode=strict,
            use_llm=use_llm,
            allow_first_name_only_match=allow_first_name_only,
        )

        if output_format == "json":
            display_json_results(results)
        else:
            display_pretty_results(results)

    except Exception as e:
        console.print(f"[red]Error processing request: {e}[/red]")
        logger.exception("Processing error")
        raise typer.Exit(1)


def process_match_request(
    name: str,
    article_text: str,
    language: str = "en",
    strict_mode: bool = False,
    use_llm: bool = False,
    allow_first_name_only_match: bool = True,
) -> List[MatchResult]:
    extractor = EntityExtractor()
    matcher = NameMatcher(allow_first_name_only_match=allow_first_name_only_match)

    console.print("[blue]Extracting entities from article...[/blue]")
    entities = extractor.extract_entities(article_text, language)

    if not entities:
        console.print("[yellow]No person entities found in article[/yellow]")
        return []

    console.print(f"[green]Found {len(entities)} person entities[/green]")

    results = []
    for entity in entities:
        console.print(f"[blue]Matching against: {entity.text}[/blue]")
        if use_llm:
            console.print("[blue]  Using LLM enhancement...[/blue]")
        result = matcher.match(name, entity, strict_mode, use_llm)
        results.append(result)

    return results


def display_pretty_results(results: List[MatchResult]):
    if not results:
        console.print("[yellow]No entities to match against[/yellow]")
        return

    results.sort(key=lambda r: r.match_confidence, reverse=True)
    
    has_warnings = any(
        ("WARNING" in result.final_decision_logic or 
         "ANALYST REVIEW" in result.final_decision_logic or
         any("WARNING" in exp.reasoning for exp in result.explanation))
        for result in results if result.is_match_recommendation
    )
    
    console.print("\n[bold]Overall Status:[/bold]")
    if not any(r.is_match_recommendation for r in results):
        console.print("✓ [green]No matches found - CLEAR[/green]\n")
    elif has_warnings:
        console.print("✗ [red]Matches found with warnings - ANALYST REVIEW REQUIRED[/red]\n")
    else:
        console.print("✓ [green]Clean matches found - NO WARNINGS[/green]\n")

    table = Table(title="Match Results Summary")
    table.add_column("Entity", style="cyan")
    table.add_column("Confidence", style="magenta")
    table.add_column("Recommendation", style="green")
    table.add_column("Decision", style="yellow")

    for result in results:
        recommendation = "MATCH" if result.is_match_recommendation else "NO MATCH"
        confidence_pct = f"{result.match_confidence:.1%}"

        table.add_row(
            result.article_entity,
            confidence_pct,
            recommendation,
            result.confidence_level.value,
        )

    console.print(table)

    console.print("\n[bold]Detailed Results:[/bold]\n")

    for result in results:
        if result.is_match_recommendation:
            content = f"""
[bold]Input Name:[/bold] {result.input_name}
[bold]Article Entity:[/bold] {result.article_entity}
[bold]Match Confidence:[/bold] {result.match_confidence:.1%}
[bold]Recommendation:[/bold] {"MATCH" if result.is_match_recommendation else "NO MATCH"}

[bold]Explanation:[/bold]
"""
            for exp in result.explanation:
                content += (
                    f"• {exp.reasoning} ({exp.confidence_score:.1%} confidence)\n"
                )

            content += f"\n[bold]Final Decision:[/bold] {result.final_decision_logic}"

            has_warning = ("WARNING" in result.final_decision_logic or 
                          "ANALYST REVIEW" in result.final_decision_logic or
                          any("WARNING" in exp.reasoning for exp in result.explanation))
            
            status_icon = "✗" if has_warning else "✓"
            status_color = "red" if has_warning else "green"
            content = f"[{status_color}]{status_icon}[/{status_color}] " + content
            
            panel = Panel(
                content,
                title=f"Match: {result.article_entity}" + (" ⚠️ REVIEW REQUIRED" if has_warning else ""),
                border_style="red" if has_warning else ("green" if result.match_confidence > 0.8 else "yellow"),
            )
            console.print(panel)


def display_json_results(results: List[MatchResult]):
    json_results = []

    for result in results:
        result_dict = result.model_dump()

        result_dict["explanation"] = [exp.reasoning for exp in result.explanation]

        json_results.append(result_dict)

    console.print(JSON(json.dumps(json_results, indent=2)))


@app.command()
def download_model(
    language: str = typer.Option("en", "--lang", "-l", help="Two-letter language code"),
):
    extractor = EntityExtractor()

    console.print(f"[blue]Downloading spaCy model for language: {language}[/blue]")

    if extractor.download_model(language):
        console.print("[green]Model downloaded successfully![/green]")
    else:
        console.print("[red]Failed to download model[/red]")
        raise typer.Exit(1)


@app.command()
def test():
    test_name = "William Clinton"
    test_article = """
    Former President Bill Clinton announced a new initiative today.
    Clinton, who served as the 42nd president, spoke at the conference.
    Also present was James Smith and Elizabeth Warren.
    """

    console.print("[blue]Running test match...[/blue]")
    console.print(f"[yellow]Name:[/yellow] {test_name}")
    console.print(f"[yellow]Article:[/yellow] {test_article[:100]}...")

    try:
        results = process_match_request(
            name=test_name, article_text=test_article, language="en", strict_mode=False
        )

        display_pretty_results(results)

        console.print("\n[green]✓ Test completed successfully![/green]")

    except Exception as e:
        console.print(f"\n[red]✗ Test failed: {e}[/red]")
        raise typer.Exit(1)


def main():
    app()


if __name__ == "__main__":
    main()