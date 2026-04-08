from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, filename, stats, video_name="Unknown"):
        filepath = str(self.output_dir / filename)
        c = canvas.Canvas(filepath, pagesize=letter)
        width, height = letter

        sport = stats.get("sport", "Football")
        action_data = stats.get("action_counts", {})
        analysis_data = stats.get("analysis_data", {})
        total_frames = stats.get("total_frames", 0)

        c.setFillColorRGB(0.1, 0.1, 0.2)
        c.rect(0, height - 100, width, 100, fill=1)

        c.setFillColorRGB(0.34, 0.65, 1.0)
        c.setFont("Helvetica-Bold", 26)
        c.drawString(40, height - 45, f"GAIze-Sport: {sport} Analysis Report")

        c.setFillColorRGB(0.8, 0.8, 0.8)
        c.setFont("Helvetica", 12)
        c.drawString(40, height - 70, f"Source Video: {video_name}")

        y = height - 140

        c.setFillColorRGB(0.1, 0.1, 0.2)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, y, "Match Summary")
        c.line(40, y - 10, width - 40, y - 10)
        y -= 30

        c.setFont("Helvetica", 12)
        c.drawString(40, y, f"Total Frames Processed: {total_frames}")
        y -= 20

        if total_frames > 0 and action_data:
            dominant = max(action_data, key=action_data.get)
            dom_pct = int(action_data[dominant] / total_frames * 100)
            c.drawString(40, y, f"Dominant Action: {dominant} ({dom_pct}%)")
            y -= 40

            actions = list(action_data.keys())
            if len(actions) >= 3:
                action_colors = {actions[0]: "#2cc985", actions[1]: "#8ab4f8", actions[2]: "#e06c75"}
                action_map = {actions[0]: 0, actions[1]: 1, actions[2]: 2, "---": -0.5}
            else:
                action_colors = {}
                action_map = {"---": -0.5}

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
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_list, startangle=140, textprops={'fontsize': 8})
                ax1.set_title("Action Distribution", fontsize=10)

            pie_path = tempfile.mktemp(suffix=".png")
            fig1.tight_layout()
            fig1.savefig(pie_path, dpi=150)
            plt.close(fig1)

            fig3, ax3 = plt.subplots(figsize=(8, 2.5))
            if analysis_data:
                sorted_keys = sorted(analysis_data.keys())
                xs = sorted_keys
                ys = [action_map.get(analysis_data[k].get("action", "---"), -0.5) for k in xs]
                cs = [action_colors.get(analysis_data[k].get("action", "---"), "#444444") for k in xs]

                ax3.scatter(xs, ys, c=cs, s=6, alpha=0.8)
                if len(actions) >= 3:
                    ax3.set_yticks([0, 1, 2])
                    ax3.set_yticklabels([a[:6] for a in actions], fontsize=8)
                ax3.set_xlabel("Frame Number", fontsize=8)
                ax3.set_title("Action Timeline", fontsize=10)
                ax3.set_ylim(-1, 3)
                ax3.spines["top"].set_visible(False)
                ax3.spines["right"].set_visible(False)

            timeline_path = tempfile.mktemp(suffix=".png")
            fig3.tight_layout()
            fig3.savefig(timeline_path, dpi=150)
            plt.close(fig3)

            if os.path.exists(pie_path):
                c.drawImage(pie_path, 40, y - 200, width=240, height=180)
                os.remove(pie_path)

            list_x = 320
            list_y = y - 30
            c.setFont("Helvetica-Bold", 14)
            c.drawString(list_x, list_y, "Action Breakdown")
            c.line(list_x, list_y - 5, list_x + 200, list_y - 5)
            list_y -= 25
            c.setFont("Helvetica", 12)
            for action, count in action_data.items():
                pct = int((count / total_frames) * 100) if total_frames > 0 else 0
                c.drawString(list_x, list_y, f"• {action}: {count} frames ({pct}%)")
                list_y -= 20

            y -= 220

            if os.path.exists(timeline_path):
                c.drawImage(timeline_path, 40, y - 180, width=480, height=150)
                os.remove(timeline_path)
                y -= 200
        else:
            c.setFillColorRGB(0.4, 0.4, 0.4)
            c.setFont("Helvetica", 12)
            c.drawString(40, y, "No action data available for this session.")
            y -= 40

        y -= 20
        if y > 100:
            c.setFillColorRGB(0.1, 0.1, 0.2)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, "AI Assessment Summary")
            y -= 20
            c.setFont("Helvetica", 11)

            avg_det = 0
            if analysis_data:
                avg_det = sum(d["detections"] for d in analysis_data.values()) / max(len(analysis_data), 1)

            c.drawString(40, y, f"• Average detected persons per frame: {avg_det:.1f}")
            y -= 15

            if action_data and total_frames > 0:
                actions = list(action_data.keys())
                if len(actions) >= 3:
                    pcts = {k: int(v / total_frames * 100) for k, v in action_data.items()}
                    if pcts.get(actions[1], 0) > 40:
                        c.drawString(40, y, f"• High {actions[1].lower()} activity detected — intense game.")
                        y -= 15
                    elif pcts.get(actions[0], 0) > 60:
                        c.drawString(40, y, f"• Mostly {actions[0].lower()} activity — limited player movement.")
                        y -= 15
                    else:
                        c.drawString(40, y, "• Balanced activity mix — diverse movement patterns.")
                        y -= 15

                    if pcts.get(actions[2], 0) > 20:
                        c.drawString(40, y, f"• Significant {actions[2].lower()} activity — key moments captured.")
                        y -= 15

        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(40, 40, "Generated by GAIze-Sport — AI Sports Video Analysis engine.")

        c.save()
        print(f"Report generated: {filepath}")
        return filepath
