from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
import os
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette that scales to any number of actions
DEFAULT_PALETTE = ["#2cc985", "#8ab4f8", "#e06c75", "#f0c040", "#bc8cff",
                   "#ff9f43", "#ee5a24", "#7ed6df", "#686de0", "#badc58"]


class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_color_maps(actions):
        """Return (action_colors, action_map) for *any* number of actions."""
        action_colors = {a: DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
                         for i, a in enumerate(actions)}
        action_colors["---"] = "#444444"

        action_map = {a: i for i, a in enumerate(actions)}
        action_map["---"] = -0.5
        return action_colors, action_map

    @staticmethod
    def _wrapped_text(c, text, x, y, max_width, font_name="Helvetica", font_size=11, leading=14):
        """Draw wrapped text on the canvas and return the new y position."""
        c.setFont(font_name, font_size)
        lines = simpleSplit(text, font_name, font_size, max_width)
        for line in lines:
            if y < 60:
                c.showPage()
                y = letter[1] - 50
                c.setFont(font_name, font_size)
            c.drawString(x, y, line)
            y -= leading
        return y

    # ------------------------------------------------------------------
    # AI Summary generation (rule-based, no external API)
    # ------------------------------------------------------------------
    def _generate_ai_summary(self, stats):
        """Build a multi-paragraph AI-style textual summary of the match."""
        sport = stats.get("sport", "Football")
        action_data = stats.get("action_counts", {})
        analysis_data = stats.get("analysis_data", {})
        total_frames = stats.get("total_frames", 0)

        paragraphs = []

        # -- Overview --
        if total_frames > 0:
            avg_det = (sum(d["detections"] for d in analysis_data.values())
                       / max(len(analysis_data), 1)) if analysis_data else 0
            paragraphs.append(
                f"This report summarises the AI-powered analysis of a {sport} video clip "
                f"spanning {total_frames} frames.  On average, {avg_det:.1f} person(s) were "
                f"detected per frame using YOLOv11 pose estimation, with actions classified "
                f"by a 4-class LSTM model."
            )
        else:
            paragraphs.append("No frames were processed; the report contains no analysis data.")
            return paragraphs

        # -- Action distribution narrative --
        total_actions = sum(action_data.values())
        if total_actions > 0:
            sorted_actions = sorted(action_data.items(), key=lambda x: x[1], reverse=True)
            dominant, dom_count = sorted_actions[0]
            dom_pct = int(dom_count / total_actions * 100)

            breakdown_parts = [f"{name} ({int(cnt / total_actions * 100)}%)"
                               for name, cnt in sorted_actions]
            paragraphs.append(
                f"The dominant action detected was '{dominant}', accounting for {dom_pct}% of all "
                f"classified frames.  Full action distribution: {', '.join(breakdown_parts)}."
            )

            # Intensity assessment
            idle_pct = int(action_data.get("Idle", 0) / total_actions * 100)
            non_idle_pct = 100 - idle_pct

            if non_idle_pct >= 70:
                paragraphs.append(
                    f"Activity intensity is HIGH — {non_idle_pct}% of frames show active player movement.  "
                    "This suggests a fast-paced, high-energy segment."
                )
            elif non_idle_pct >= 40:
                paragraphs.append(
                    f"Activity intensity is MODERATE — {non_idle_pct}% of frames contain active movement "
                    "with regular idle periods, indicating balanced play."
                )
            else:
                paragraphs.append(
                    f"Activity intensity is LOW — only {non_idle_pct}% of frames show active movement.  "
                    "The footage may capture a stoppage, warm-up, or low-tempo phase."
                )

            # Per-action insights (all actions, not just 3)
            for action_name, count in sorted_actions:
                pct = int(count / total_actions * 100)
                if action_name == "Idle":
                    continue
                if pct >= 30:
                    paragraphs.append(
                        f"Notable: '{action_name}' is a major component at {pct}%, "
                        "suggesting significant focus on this activity during the clip."
                    )
                elif pct >= 15:
                    paragraphs.append(
                        f"'{action_name}' occurs at a meaningful rate of {pct}%, "
                        "contributing to the overall action complexity."
                    )

        # -- Temporal patterns --
        if analysis_data:
            total = len(analysis_data)
            midpoint = total // 2
            sorted_keys = sorted(analysis_data.keys())
            first_half = sorted_keys[:midpoint]
            second_half = sorted_keys[midpoint:]

            def _active_ratio(keys):
                active = sum(1 for k in keys if analysis_data[k].get("action", "---") != "Idle"
                             and analysis_data[k].get("action", "---") != "---")
                return active / max(len(keys), 1)

            r1, r2 = _active_ratio(first_half), _active_ratio(second_half)
            if r2 > r1 * 1.3:
                paragraphs.append(
                    "Temporal analysis: activity picks up in the second half of the clip, "
                    "indicating an increase in game tempo over time."
                )
            elif r1 > r2 * 1.3:
                paragraphs.append(
                    "Temporal analysis: the first half of the clip is more active; "
                    "the second half shows reduced movement."
                )
            else:
                paragraphs.append(
                    "Temporal analysis: activity is relatively consistent throughout the clip."
                )

        # -- Detection density --
        if analysis_data:
            det_values = [d["detections"] for d in analysis_data.values()]
            max_det = max(det_values)
            min_det = min(det_values)
            avg_det_overall = sum(det_values) / len(det_values)
            paragraphs.append(
                f"Player detection density ranges from {min_det} to {max_det} per frame "
                f"(average {avg_det_overall:.1f}).  "
                + ("Dense scenes provide richer tactical data." if avg_det_overall >= 4
                   else "Relatively sparse detections — consider a wider camera angle for higher coverage.")
            )

        paragraphs.append(
            "This analysis was generated automatically by the GAIze-Sport AI engine.  "
            "Results depend on model accuracy and video quality."
        )
        return paragraphs

    # ------------------------------------------------------------------
    # main report
    # ------------------------------------------------------------------
    def generate_report(self, filename, stats, video_name="Unknown"):
        filepath = str(self.output_dir / filename)
        c = canvas.Canvas(filepath, pagesize=letter)
        width, height = letter

        sport = stats.get("sport", "Football")
        action_data = stats.get("action_counts", {})
        analysis_data = stats.get("analysis_data", {})
        total_frames = stats.get("total_frames", 0)

        # ── Header ────────────────────────────────────────────────────
        c.setFillColorRGB(0.1, 0.1, 0.2)
        c.rect(0, height - 100, width, 100, fill=1)

        c.setFillColorRGB(0.34, 0.65, 1.0)
        c.setFont("Helvetica-Bold", 26)
        c.drawString(40, height - 45, f"GAIze-Sport: {sport} Analysis Report")

        c.setFillColorRGB(0.8, 0.8, 0.8)
        c.setFont("Helvetica", 12)
        c.drawString(40, height - 70, f"Source Video: {video_name}")

        y = height - 140

        # ── Match Summary ─────────────────────────────────────────────
        c.setFillColorRGB(0.1, 0.1, 0.2)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, y, "Match Summary")
        c.setStrokeColorRGB(0.34, 0.65, 1.0)
        c.line(40, y - 10, width - 40, y - 10)
        y -= 30

        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 12)
        c.drawString(40, y, f"Total Frames Processed: {total_frames}")
        y -= 20

        actions = list(action_data.keys())
        num_actions = len(actions)

        if total_frames > 0 and action_data:
            dominant = max(action_data, key=action_data.get)
            dom_pct = int(action_data[dominant] / total_frames * 100)
            c.drawString(40, y, f"Dominant Action: {dominant} ({dom_pct}%)")
            y -= 20

            total_actions_sum = sum(action_data.values())
            active_count = sum(v for k, v in action_data.items() if k != "Idle")
            active_pct = int(active_count / total_actions_sum * 100) if total_actions_sum > 0 else 0
            c.drawString(40, y, f"Activity Rate (non-Idle): {active_pct}%")
            y -= 20

            avg_det = 0
            if analysis_data:
                avg_det = sum(d["detections"] for d in analysis_data.values()) / max(len(analysis_data), 1)
            c.drawString(40, y, f"Avg Detected Persons / Frame: {avg_det:.1f}")
            y -= 30

            # -- build colour maps dynamically --
            action_colors, action_map = self._build_color_maps(actions)

            # ── Pie Chart ─────────────────────────────────────────────
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            labels = []
            sizes = []
            colors_list = []
            for k, v in action_data.items():
                if v > 0:
                    labels.append(k)
                    sizes.append(v)
                    colors_list.append(action_colors.get(k, "#888888"))

            if sizes:
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_list,
                        startangle=140, textprops={'fontsize': 8})
                ax1.set_title("Action Distribution", fontsize=10)

            pie_path = tempfile.mktemp(suffix=".png")
            fig1.tight_layout()
            fig1.savefig(pie_path, dpi=150)
            plt.close(fig1)

            # ── Timeline Chart ────────────────────────────────────────
            fig3, ax3 = plt.subplots(figsize=(8, 2.5))
            if analysis_data:
                sorted_keys = sorted(analysis_data.keys())
                xs = sorted_keys
                ys = [action_map.get(analysis_data[k].get("action", "---"), -0.5) for k in xs]
                cs = [action_colors.get(analysis_data[k].get("action", "---"), "#444444") for k in xs]

                ax3.scatter(xs, ys, c=cs, s=6, alpha=0.8)
                ax3.set_yticks(list(range(num_actions)))
                ax3.set_yticklabels([a[:8] for a in actions], fontsize=8)
                ax3.set_xlabel("Frame Number", fontsize=8)
                ax3.set_title("Action Timeline", fontsize=10)
                ax3.set_ylim(-1, num_actions)
                ax3.spines["top"].set_visible(False)
                ax3.spines["right"].set_visible(False)

            timeline_path = tempfile.mktemp(suffix=".png")
            fig3.tight_layout()
            fig3.savefig(timeline_path, dpi=150)
            plt.close(fig3)

            # ── Draw Pie + Breakdown side-by-side ─────────────────────
            if os.path.exists(pie_path):
                c.drawImage(pie_path, 40, y - 200, width=240, height=180)
                os.remove(pie_path)

            list_x = 320
            list_y = y - 30
            c.setFillColorRGB(0.1, 0.1, 0.2)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(list_x, list_y, "Action Breakdown")
            c.setStrokeColorRGB(0.34, 0.65, 1.0)
            c.line(list_x, list_y - 5, list_x + 200, list_y - 5)
            list_y -= 25
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica", 12)
            for action, count in action_data.items():
                pct = int((count / total_frames) * 100) if total_frames > 0 else 0
                c.drawString(list_x, list_y, f"• {action}: {count} frames ({pct}%)")
                list_y -= 20

            y -= 220

            # ── Timeline image ────────────────────────────────────────
            if os.path.exists(timeline_path):
                c.drawImage(timeline_path, 40, y - 180, width=480, height=150)
                os.remove(timeline_path)
                y -= 200
        else:
            c.setFillColorRGB(0.4, 0.4, 0.4)
            c.setFont("Helvetica", 12)
            c.drawString(40, y, "No action data available for this session.")
            y -= 40

        # ── AI Assessment Summary (quick bullets) ─────────────────────
        y -= 20
        if y < 120:
            c.showPage()
            y = height - 50

        c.setFillColorRGB(0.1, 0.1, 0.2)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, "AI Assessment Summary")
        c.setStrokeColorRGB(0.34, 0.65, 1.0)
        c.line(40, y - 5, width - 40, y - 5)
        y -= 22
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 11)

        avg_det = 0
        if analysis_data:
            avg_det = sum(d["detections"] for d in analysis_data.values()) / max(len(analysis_data), 1)

        c.drawString(40, y, f"• Average detected persons per frame: {avg_det:.1f}")
        y -= 18

        if action_data and total_frames > 0:
            total_act = sum(action_data.values())
            if total_act > 0:
                pcts = {k: int(v / total_act * 100) for k, v in action_data.items()}

                # List ALL action percentages in the breakdown line
                breakdown_str = ", ".join(f"{a} {pcts[a]}%" for a in actions)
                y = self._wrapped_text(c, f"• Action breakdown: {breakdown_str}.",
                                       40, y, width - 80, "Helvetica", 11, 15)

                idle_pct = pcts.get("Idle", 0)
                non_idle_pct = 100 - idle_pct

                if non_idle_pct >= 60:
                    c.drawString(40, y, f"• High activity ({non_idle_pct}% non-idle) — intense, fast-paced game.")
                elif non_idle_pct >= 30:
                    c.drawString(40, y, f"• Moderate activity ({non_idle_pct}% non-idle) — balanced play.")
                else:
                    c.drawString(40, y, f"• Low activity ({non_idle_pct}% non-idle) — limited movement.")
                y -= 18

                # Mention every notable non-Idle action (not just index 2)
                for i, action in enumerate(actions):
                    if action == "Idle":
                        continue
                    if pcts.get(action, 0) > 20:
                        c.drawString(40, y, f"• Significant {action.lower()} activity ({pcts[action]}%) — key moments captured.")
                        y -= 18

        # ── AI-Generated Summary (new section, multi-paragraph) ───────
        y -= 15
        if y < 120:
            c.showPage()
            y = height - 50

        c.setFillColorRGB(0.1, 0.1, 0.2)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, "AI-Generated Match Summary")
        c.setStrokeColorRGB(0.96, 0.75, 0.25)
        c.line(40, y - 8, width - 40, y - 8)
        y -= 28

        ai_paragraphs = self._generate_ai_summary(stats)
        c.setFillColorRGB(0.15, 0.15, 0.15)
        for para in ai_paragraphs:
            y = self._wrapped_text(c, para, 50, y, width - 100, "Helvetica", 10.5, 14)
            y -= 8  # paragraph spacing
            if y < 60:
                c.showPage()
                y = height - 50

        # ── Footer ────────────────────────────────────────────────────
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(40, 40, "Generated by GAIze-Sport — AI Sports Video Analysis engine.")

        c.save()
        print(f"Report generated: {filepath}")
        return filepath
