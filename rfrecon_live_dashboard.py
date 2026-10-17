
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import iio
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from rfrecon_consistent_features import (
    extract_features,
    FEATURE_NAMES,
)

# ============================================================
# RFRecon configuration
# ============================================================

MODEL_FILE = "rfrecon_four_class_best_rf.pkl"
FEATURE_FILE = "rfrecon_four_class_best_features.pkl"

WINDOW_SIZE = 4096
PLUTO_URI = "ip:192.168.2.1"
CAPTURE_DIR = Path("real_dataset")

CLASS_NAMES = {
    "noise": "Noise",
    "wifi_like": "WiFi-like",
    "bluetooth_like": "Bluetooth-like",
    "cellular_like": "Cellular-like",
}


class RFReconDashboard:

    def __init__(self, root):
        self.root = root
        self.root.title("RFRecon | RF Signal Classification")
        self.root.geometry("1250x940")
        self.root.minsize(950, 720)

        self.capture_path = None
        self.results = None
        self.iq = None
        self.current_sample_rate = None

        self.sample_rate = tk.StringVar(value="30.72")
        self.status = tk.StringVar(
            value="Select a saved capture or capture from Pluto."
        )

        try:
            self.model = joblib.load(MODEL_FILE)
            self.features = joblib.load(FEATURE_FILE)

            if self.features != FEATURE_NAMES:
                raise ValueError(
                    "Model feature order does not match the extractor."
                )

            if set(self.model.classes_) != set(CLASS_NAMES):
                raise ValueError(
                    f"Unexpected model classes: {list(self.model.classes_)}"
                )

        except Exception as exc:
            messagebox.showerror("Model loading error", str(exc))
            self.root.after(100, self.root.destroy)
            return

        self.build_ui()
        self.draw_empty_charts()

    # ========================================================
    # User interface
    # ========================================================

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="RFRecon",
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text=(
                "Four-class RF reconnaissance | "
                "ADALM-PLUTO IQ analysis"
            ),
        ).pack(anchor="w", pady=(0, 10))

        controls = ttk.LabelFrame(
            outer, text="Capture settings", padding=10
        )
        controls.pack(fill="x", pady=4)

        ttk.Label(
            controls, text="Sample rate (MS/s):"
        ).grid(row=0, column=0, padx=4, sticky="w")

        ttk.Entry(
            controls,
            textvariable=self.sample_rate,
            width=10,
        ).grid(row=0, column=1, padx=4)

        ttk.Button(
            controls,
            text="Select IQ capture",
            command=self.select_capture,
        ).grid(row=0, column=2, padx=4)

        ttk.Button(
            controls,
            text="Analyze capture",
            command=self.analyze,
        ).grid(row=0, column=3, padx=4)

        ttk.Button(
            controls,
            text="Capture from Pluto",
            command=self.capture_from_pluto,
        ).grid(row=0, column=4, padx=4)

        self.file_label = ttk.Label(
            outer,
            text="No capture selected",
            wraplength=1180,
        )
        self.file_label.pack(anchor="w", pady=5)

        # Prediction summary
        summary = ttk.LabelFrame(
            outer, text="Prediction summary", padding=10
        )
        summary.pack(fill="x", pady=4)

        self.prediction_label = ttk.Label(
            summary,
            text="Dominant predicted class: —",
            font=("Segoe UI", 15, "bold"),
        )
        self.prediction_label.pack(anchor="w")

        self.probability_label = ttk.Label(
            summary,
            text="Mean maximum model probability: —",
        )
        self.probability_label.pack(anchor="w", pady=2)

        self.info_label = ttk.Label(
            summary,
            text="Samples: — | Windows: — | Sample rate: —",
        )
        self.info_label.pack(anchor="w", pady=2)

        # Spectrum and probability charts
        chart_frame = ttk.Frame(outer)
        chart_frame.pack(fill="both", expand=True, pady=5)

        self.figure = Figure(figsize=(10, 6), dpi=85)
        self.spectrum_ax = self.figure.add_subplot(211)
        self.probability_ax = self.figure.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(
            self.figure, master=chart_frame
        )
        self.canvas.get_tk_widget().pack(
            fill="both", expand=True
        )

        # Per-class summary table
        table_frame = ttk.LabelFrame(
            outer, text="Per-class prediction summary", padding=5
        )
        table_frame.pack(fill="x", pady=4)

        columns = (
            "class",
            "count",
            "share",
            "mean_probability",
        )

        self.table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=4,
        )

        self.table.heading("class", text="Predicted class")
        self.table.heading("count", text="Windows")
        self.table.heading("share", text="Share (%)")
        self.table.heading(
            "mean_probability",
            text="Mean max probability (%)",
        )

        self.table.column("class", width=180, anchor="w")
        self.table.column("count", width=100, anchor="center")
        self.table.column("share", width=120, anchor="center")
        self.table.column(
            "mean_probability", width=230, anchor="center"
        )

        self.table.pack(fill="x")

        # Export actions
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=5)

        ttk.Button(
            actions,
            text="Export predictions CSV",
            command=self.export_results,
        ).pack(side="right", padx=4)

        ttk.Button(
            actions,
            text="Export report bundle",
            command=self.export_report_bundle,
        ).pack(side="right", padx=4)

        ttk.Label(
            outer,
            textvariable=self.status,
            wraplength=1180,
        ).pack(anchor="w", pady=3)

        ttk.Label(
            outer,
            text=(
                "Research prototype: model probabilities are not "
                "verified protocol identities. Results require validation "
                "against labelled captures."
            ),
            wraplength=1180,
        ).pack(anchor="w", pady=(3, 0))

    # ========================================================
    # Charts
    # ========================================================

    def draw_empty_charts(self):
        self.spectrum_ax.clear()
        self.spectrum_ax.set_title("RF Spectrum")
        self.spectrum_ax.set_xlabel("Frequency offset (MHz)")
        self.spectrum_ax.set_ylabel("Relative power (dB)")
        self.spectrum_ax.grid(True, alpha=0.3)

        self.probability_ax.clear()
        self.probability_ax.set_title("Average Model Probabilities")
        self.probability_ax.set_ylabel("Probability (%)")
        self.probability_ax.set_ylim(0, 100)
        self.probability_ax.grid(axis="y", alpha=0.3)

        self.figure.tight_layout(pad=2.5)
        self.canvas.draw_idle()

    def update_spectrum(self, iq, sample_rate):
        window = iq[:WINDOW_SIZE]
        windowed_iq = window * np.hanning(len(window))

        spectrum = np.fft.fftshift(
            np.fft.fft(windowed_iq)
        )
        power = np.abs(spectrum) ** 2

        frequencies = np.fft.fftshift(
            np.fft.fftfreq(
                len(window),
                d=1.0 / sample_rate,
            )
        )

        relative_db = 10 * np.log10(
            power / (np.max(power) + 1e-20) + 1e-12
        )

        self.spectrum_ax.clear()
        self.spectrum_ax.plot(
            frequencies / 1e6,
            relative_db,
            linewidth=0.8,
        )
        self.spectrum_ax.set_title(
            "RF Spectrum — First 4096-sample window"
        )
        self.spectrum_ax.set_xlabel("Frequency offset (MHz)")
        self.spectrum_ax.set_ylabel("Relative power (dB)")
        self.spectrum_ax.set_xlim(
            -sample_rate / 2e6,
            sample_rate / 2e6,
        )
        self.spectrum_ax.grid(True, alpha=0.3)

    def update_probability_chart(self, probabilities):
        average = probabilities.mean(axis=0) * 100

        labels = [
            CLASS_NAMES[name] for name in self.model.classes_
        ]

        self.probability_ax.clear()
        bars = self.probability_ax.bar(labels, average)

        self.probability_ax.set_title(
            "Average Model Probabilities Across Windows"
        )
        self.probability_ax.set_ylabel("Probability (%)")
        self.probability_ax.set_ylim(0, 100)
        self.probability_ax.grid(axis="y", alpha=0.3)
        self.probability_ax.tick_params(
            axis="x", labelrotation=10
        )

        for bar, value in zip(bars, average):
            self.probability_ax.annotate(
                f"{value:.1f}%",
                (
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        self.figure.tight_layout(pad=2.5)
        self.canvas.draw_idle()

    # ========================================================
    # File selection
    # ========================================================

    def select_capture(self):
        path = filedialog.askopenfilename(
            title="Select complex64 IQ capture",
            filetypes=[
                ("IQ capture files", "*.dat *.bin *.c64"),
                ("All files", "*.*"),
            ],
        )

        if path:
            self.capture_path = Path(path)
            self.file_label.config(
                text=f"Capture: {self.capture_path}"
            )
            self.results = None
            self.iq = None
            self.status.set("Capture selected. Ready to analyze.")

    # ========================================================
    # Direct Pluto capture
    # ========================================================

    def capture_from_pluto(self):
        samples = 262144
        output_dir = CAPTURE_DIR

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / f"pluto_capture_{timestamp}.dat"

        buf = None
        i_ch = None
        q_ch = None
        channels_enabled = False

        try:
            self.status.set("Connecting to Pluto SDR...")
            self.root.update_idletasks()

            ctx = iio.Context(PLUTO_URI)
            rx = ctx.find_device("cf-ad9361-lpc")

            if rx is None:
                raise RuntimeError("Pluto RX data device not found.")

            channels = {ch.id: ch for ch in rx.channels}

            if "voltage0" not in channels or "voltage1" not in channels:
                raise RuntimeError(
                    f"Expected I/Q channels; found {list(channels)}"
                )

            i_ch = channels["voltage0"]
            q_ch = channels["voltage1"]

            i_ch.enabled = True
            q_ch.enabled = True
            channels_enabled = True

            self.status.set("Capturing IQ samples from Pluto...")
            self.root.update_idletasks()

            buf = iio.Buffer(rx, samples, False)
            buf.refill()
            raw = buf.read()

            expected_bytes = samples * 4

            if len(raw) != expected_bytes:
                raise RuntimeError(
                    f"Received {len(raw)} bytes; expected "
                    f"{expected_bytes}. Capture was not saved."
                )

            iq16 = np.frombuffer(raw, dtype="<i2")

            iq = (
                iq16[0::2].astype(np.float32)
                + 1j * iq16[1::2].astype(np.float32)
            ) / 32768.0

            if not (
                np.isfinite(iq.real).all()
                and np.isfinite(iq.imag).all()
            ):
                raise RuntimeError("Non-finite IQ samples detected.")

            output_dir.mkdir(parents=True, exist_ok=True)
            iq.astype(np.complex64).tofile(output_path)

        except Exception as exc:
            messagebox.showerror(
                "Pluto capture error",
                f"{exc}\n\nCheck the connection and try again.",
            )
            self.status.set("Pluto capture failed.")
            return

        finally:
            if buf is not None:
                del buf

            if channels_enabled:
                try:
                    i_ch.enabled = False
                    q_ch.enabled = False
                except Exception:
                    pass

        self.capture_path = output_path
        self.file_label.config(text=f"Capture: {output_path}")
        self.status.set(f"Capture saved: {output_path}")

        messagebox.showinfo(
            "Capture complete",
            f"Saved {samples:,} complex IQ samples:\n{output_path}",
        )

        self.analyze()

    # ========================================================
    # Feature extraction and classification
    # ========================================================

    def analyze(self):
        if self.capture_path is None:
            messagebox.showwarning(
                "Select a capture",
                "Choose a saved capture or capture from Pluto first.",
            )
            return

        try:
            fs_msps = float(self.sample_rate.get())

            if not np.isfinite(fs_msps) or fs_msps <= 0:
                raise ValueError(
                    "Sample rate must be a positive number."
                )

            sample_rate = int(fs_msps * 1_000_000)

            self.status.set("Loading IQ data...")
            self.root.update_idletasks()

            iq = np.fromfile(
                self.capture_path,
                dtype=np.complex64,
            )

            if iq.size == 0:
                raise ValueError("Capture file is empty.")

            if not (
                np.isfinite(iq.real).all()
                and np.isfinite(iq.imag).all()
            ):
                raise ValueError("Capture contains non-finite IQ samples.")

            count = len(iq) // WINDOW_SIZE

            if count == 0:
                raise ValueError(
                    f"Capture needs at least {WINDOW_SIZE} samples."
                )

            self.status.set(f"Extracting features from {count} windows...")
            self.root.update_idletasks()

            feature_rows = []

            for index in range(count):
                start = index * WINDOW_SIZE
                window = iq[start:start + WINDOW_SIZE]

                feature_rows.append(
                    extract_features(window, sample_rate)
                )

            X = pd.DataFrame(
                feature_rows,
                columns=self.features,
            )

            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            result = pd.DataFrame({
                "window": np.arange(count),
                "predicted_class": predictions,
                "max_model_probability": probabilities.max(axis=1),
            })

            for index, name in enumerate(self.model.classes_):
                result[f"prob_{name}"] = probabilities[:, index]

            self.results = result
            self.iq = iq
            self.current_sample_rate = sample_rate

            counts = result["predicted_class"].value_counts()
            dominant = counts.index[0]
            mean_probability = result[
                "max_model_probability"
            ].mean()

            self.prediction_label.config(
                text=f"Dominant predicted class: {CLASS_NAMES[dominant]}"
            )
            self.probability_label.config(
                text=(
                    "Mean maximum model probability: "
                    f"{mean_probability:.2%}"
                )
            )
            self.info_label.config(
                text=(
                    f"Samples: {len(iq):,} | "
                    f"Windows: {count} | "
                    f"Sample rate: {fs_msps:.3f} MS/s"
                )
            )

            for item in self.table.get_children():
                self.table.delete(item)

            for name in self.model.classes_:
                subset = result[
                    result["predicted_class"] == name
                ]
                n = len(subset)

                mean_prob = (
                    f"{100 * subset['max_model_probability'].mean():.2f}"
                    if n else "—"
                )

                self.table.insert(
                    "",
                    "end",
                    values=(
                        CLASS_NAMES[name],
                        n,
                        f"{100 * n / count:.2f}",
                        mean_prob,
                    ),
                )

            self.update_spectrum(iq, sample_rate)
            self.update_probability_chart(probabilities)

            self.status.set(
                "Analysis complete. Protocol identities are not verified."
            )

        except Exception as exc:
            messagebox.showerror("Analysis error", str(exc))
            self.status.set("Analysis failed.")

    # ========================================================
    # Report bundle export
    # ========================================================

    def export_report_bundle(self):
        if self.results is None or self.iq is None:
            messagebox.showwarning(
                "No results",
                "Capture and analyze an IQ file before exporting.",
            )
            return

        if self.current_sample_rate is None:
            messagebox.showerror(
                "Missing metadata",
                "Sample-rate information is unavailable.",
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_name = f"RFRecon_report_{timestamp}"

        folder = filedialog.askdirectory(
            title="Choose a folder for the RFRecon report"
        )

        if not folder:
            return

        try:
            output_dir = Path(folder) / suggested_name
            output_dir.mkdir(parents=True, exist_ok=False)

            # 1. Window-level prediction CSV
            self.results.to_csv(
                output_dir / "predictions.csv",
                index=False,
            )

            # 2. Standalone spectrum image
            window = self.iq[:WINDOW_SIZE]
            windowed_iq = window * np.hanning(len(window))

            spectrum = np.fft.fftshift(
                np.fft.fft(windowed_iq)
            )
            power = np.abs(spectrum) ** 2

            frequencies = np.fft.fftshift(
                np.fft.fftfreq(
                    len(window),
                    d=1.0 / self.current_sample_rate,
                )
            )

            relative_db = 10 * np.log10(
                power / (np.max(power) + 1e-20) + 1e-12
            )

            report_figure = Figure(figsize=(11, 5), dpi=150)
            report_ax = report_figure.add_subplot(111)

            report_ax.plot(
                frequencies / 1e6,
                relative_db,
                linewidth=0.8,
            )
            report_ax.set_title(
                "RFRecon Spectrum — First IQ Window"
            )
            report_ax.set_xlabel("Frequency offset (MHz)")
            report_ax.set_ylabel("Relative power (dB)")
            report_ax.grid(True, alpha=0.3)
            report_figure.tight_layout()

            report_figure.savefig(
                output_dir / "spectrum.png",
                dpi=150,
                bbox_inches="tight",
            )

            # 3. Capture and prediction metadata
            counts = self.results["predicted_class"].value_counts()
            dominant = counts.index[0]

            mean_probability = self.results[
                "max_model_probability"
            ].mean()

            with (
                output_dir / "report_metadata.txt"
            ).open("w", encoding="utf-8") as report:

                report.write("RFRecon Experiment Report\n")
                report.write("=" * 40 + "\n\n")

                report.write(
                    f"Report generated: {datetime.now().isoformat()}\n"
                )
                report.write(
                    f"Capture file: {self.capture_path.resolve()}\n"
                )
                report.write(
                    f"Complex IQ samples: {len(self.iq)}\n"
                )
                report.write(
                    f"Sample rate (Hz): {self.current_sample_rate}\n"
                )
                report.write(
                    f"Window size (samples): {WINDOW_SIZE}\n"
                )
                report.write(
                    f"Complete windows analyzed: {len(self.results)}\n\n"
                )

                report.write("Prediction summary\n")
                report.write("-" * 40 + "\n")
                report.write(
                    f"Dominant predicted class: "
                    f"{CLASS_NAMES[dominant]}\n"
                )
                report.write(
                    f"Mean maximum model probability: "
                    f"{mean_probability:.4%}\n\n"
                )

                report.write("Window prediction counts\n")
                for class_name in self.model.classes_:
                    count = int(
                        (
                            self.results["predicted_class"]
                            == class_name
                        ).sum()
                    )
                    report.write(
                        f"{CLASS_NAMES[class_name]}: {count}\n"
                    )

                report.write("\nInterpretation and limitations\n")
                report.write("-" * 40 + "\n")
                report.write(
                    "Class labels are model predictions, not independently "
                    "verified protocol identities.\n"
                )
                report.write(
                    "Model probabilities are not proof of correctness or "
                    "calibrated confidence.\n"
                )
                report.write(
                    "Spectrum power is relative and is not calibrated dBm.\n"
                )
                report.write(
                    "The sample rate is the value entered in the dashboard; "
                    "verify it against the SDR configuration.\n"
                )
                report.write(
                    "Prediction accuracy requires comparison with "
                    "independently labelled test captures.\n"
                )

            messagebox.showinfo(
                "Report exported",
                f"RFRecon report saved to:\n{output_dir}\n\n"
                "Files:\n"
                "- predictions.csv\n"
                "- spectrum.png\n"
                "- report_metadata.txt",
            )

            self.status.set(f"Report exported to {output_dir}")

        except Exception as exc:
            messagebox.showerror(
                "Report export error",
                str(exc),
            )
            self.status.set("Report export failed.")

    # ========================================================
    # CSV export
    # ========================================================

    def export_results(self):
        if self.results is None:
            messagebox.showwarning(
                "No results",
                "Analyze a capture before exporting.",
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save prediction results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="rfrecon_predictions.csv",
        )

        if path:
            try:
                self.results.to_csv(path, index=False)
                self.status.set(f"Results exported to {path}")

                messagebox.showinfo(
                    "Export complete",
                    f"Saved:\n{path}",
                )

            except Exception as exc:
                messagebox.showerror(
                    "Export error",
                    str(exc),
                )


if __name__ == "__main__":
    root = tk.Tk()
    app = RFReconDashboard(root)
    root.mainloop()
