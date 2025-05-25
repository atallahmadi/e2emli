import os
import re
import sys
import threading
import tkinter as tk

from tkinter import ttk
from tkinter import filedialog, messagebox
from app_MLI import dir_score, indir_score
from app_UTILS import extract_tiles, run_eccn, reconstruct_heatmaps, extract_hm_tiles, run_accn, reconstruct_final_heatmaps, clean_temp

MASKS = {"BG.tif", "BR.tif", "BV.tif", "AVLW.tif"}


class PrintLogger:
    def __init__(self, textbox):
        self.textbox = textbox
        self.ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x08\x0A\x0D])[\x30-\x7E]*')
        self.tqdm_details_pattern = re.compile(r'\[.*?\]')

    def write(self, text):
        cleaned_text = self.ansi_escape.sub('', text)
        cleaned_text = self.tqdm_details_pattern.sub('', cleaned_text)
        if cleaned_text.strip():
            self.textbox.configure(state="normal")
            if not cleaned_text.endswith('\n'):
                cleaned_text += '\n'
            self.textbox.insert("end", cleaned_text)
            self.textbox.see("end")
            self.textbox.configure(state="disabled")

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Air Space Size Assessment")
        self.root.geometry("700x650")
        self.root.resizable(False, False)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.create_string_vars()
        self.create_dir_frame()
        self.create_ss_frame()
        self.create_mli_frame()
        self.create_end_frame()
        self.redirect_logging()

    ################# GUI FUNCTIONS #################

    def create_string_vars(self):
        self.image_name = ""
        self.file_path = ""
        self.file_path_tk = tk.StringVar()
        self.tif_files = []
        self.mask_path = tk.StringVar()

        self.model_path = tk.StringVar()
        self.folder_path = tk.StringVar()
        self.run_mode = tk.StringVar(value="Single")
        self.tile_size = tk.StringVar(value="256")
        self.tile_overlap = tk.StringVar(value="0")
        self.fov_x = tk.StringVar(value="1388")
        self.fov_y = tk.StringVar(value="1072")
        self.overlap_x = tk.StringVar(value="50")
        self.overlap_y = tk.StringVar(value="50")
        self.guidelength = tk.StringVar(value="155.3")
        self.save_fov_tf = tk.StringVar(value="0")
        self.method_dir = tk.StringVar(value="1")
        self.method_indir = tk.StringVar(value="0")
        self.score_dir = tk.StringVar(value="0.00")
        self.score_indir = tk.StringVar(value="0.00")

    def create_dir_frame(self):
        x = tk.LabelFrame(self.root, text="Directories", padx=10, pady=10)
        x.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        for i in range(3):
            x.grid_columnconfigure(i, weight=1)
        for j in range(4):
            x.grid_rowconfigure(j, weight=1)

        tk.Label(x, text="Image Directory", anchor="w").grid(
            row=0, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Entry(x, textvariable=self.file_path_tk, state="readonly").grid(
            row=0, column=1, columnspan=2, padx=10, pady=10, sticky="we"
        )

        tk.Label(x, text="Dataset Directory", anchor="w").grid(
            row=1, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Entry(x, textvariable=self.folder_path, state="readonly").grid(
            row=1, column=1, columnspan=2, padx=10, pady=10, sticky="we"
        )

        tk.Label(x, text="Selection Mode", anchor="w").grid(
            row=2, column=0, padx=10, pady=10, sticky="we"
        )
        self.combo = ttk.Combobox(
            x, values=["Single", "Entire"], textvariable=self.run_mode, state="readonly")
        self.combo.grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky="we")
        self.combo.bind("<<ComboboxSelected>>", self.collect_tifs)

        self.select_image_btn = tk.Button(x, text="Select Image", command=lambda: self.select_folder_dataset())
        self.select_image_btn.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="we")

    def create_ss_frame(self):
        x = tk.LabelFrame(self.root, text="Semantics Segmentation", padx=10, pady=10)
        x.grid(row=1, column=0, padx=10, pady=10, sticky="nswe")
        for i in range(3):
            x.grid_columnconfigure(i, weight=1)
            x.grid_rowconfigure(i, weight=1)

        tk.Label(x, text="Model Directory", anchor="w").grid(
            row=0, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Entry(x, textvariable=self.model_path, state="readonly").grid(
            row=0, column=1, columnspan=2, padx=10, pady=10, sticky="we"
        )
        self.select_model_btn = tk.Button(x, text="Select Model", command=lambda: self.select_folder_model())
        self.select_model_btn.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="we")

        self.create_masks_btn = tk.Button(x, text="Create Masks", command=lambda: self._segment_wsi())
        self.create_masks_btn.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="we")

    def create_mli_frame(self):
        x = tk.LabelFrame(
            self.root, text="Mean Linear Intercept (MLI)", padx=10, pady=10
        )
        x.grid(row=0, rowspan=2, column=1, padx=10, pady=10, sticky="nswe")

        for i in range(3):
            x.grid_columnconfigure(i, weight=1)
        for j in range(9):
            x.grid_rowconfigure(j, weight=1)

        tk.Label(x, text="FOV X (pixel)", anchor="w").grid(
            row=0, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Entry(
            x,
            textvariable=self.fov_x,
            validate="key",
            validatecommand=(self.root.register(self.validate_numeric), "%P"),
        ).grid(row=0, column=1, columnspan=2, padx=10, pady=10, sticky="we")

        tk.Label(x, text="FOV Y (pixel)", anchor="w").grid(
            row=1, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Entry(
            x,
            textvariable=self.fov_y,
            validate="key",
            validatecommand=(self.root.register(self.validate_numeric), "%P"),
        ).grid(row=1, column=1, columnspan=2, padx=10, pady=10, sticky="we")

        tk.Label(x, text="Overlap X (%)", anchor="w").grid(
            row=2, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Entry(
            x,
            textvariable=self.overlap_x,
            validate="key",
            validatecommand=(self.root.register(self.validate_numeric), "%P"),
        ).grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky="we")

        tk.Label(x, text="Overlap Y (%)", anchor="w").grid(
            row=3, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Entry(
            x,
            textvariable=self.overlap_y,
            validate="key",
            validatecommand=(self.root.register(self.validate_numeric), "%P"),
        ).grid(row=3, column=1, columnspan=2, padx=10, pady=10, sticky="we")

        tk.Label(x, text="Guideline Length (µm)", anchor="w").grid(
            row=4, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Entry(
            x,
            textvariable=self.guidelength,
            validate="key",
            validatecommand=(self.root.register(self.validate_numeric_float), "%P"),
        ).grid(row=4, column=1, columnspan=2, padx=10, pady=10, sticky="we")

        tk.Label(x, text="Save FOVs", anchor="w").grid(
            row=5, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Radiobutton(x, text="True", variable=self.save_fov_tf, value=1).grid(
            row=5, column=1, padx=10, pady=10, sticky="we"
        )
        tk.Radiobutton(x, text="False", variable=self.save_fov_tf, value=0).grid(
            row=5, column=2, padx=10, pady=10, sticky="we"
        )

        tk.Label(x, text="Calculation Method", anchor="w").grid(
            row=6, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Checkbutton(
            x, text="Direct", variable=self.method_dir, onvalue=1, offvalue=0
        ).grid(row=6, column=1, padx=10, pady=10, sticky="we")
        tk.Checkbutton(
            x, text="Indirect", variable=self.method_indir, onvalue=1, offvalue=0
        ).grid(row=6, column=2, padx=10, pady=10, sticky="we")

        tk.Label(x, text="MLI Score", anchor="w").grid(
            row=7, column=0, padx=10, pady=10, sticky="we"
        )
        tk.Label(x, textvariable=self.score_dir, anchor="w").grid(
            row=7, column=1, padx=10, pady=10, sticky="we"
        )
        tk.Label(x, textvariable=self.score_indir, anchor="w").grid(
            row=7, column=2, padx=10, pady=10, sticky="we"
        )
        self.calc_mli_btn = tk.Button(x, text="Calculate MLI", command=lambda: self._calc_mli())
        self.calc_mli_btn.grid(row=8, column=0, columnspan=3, padx=10, pady=10, sticky="we")

    def create_end_frame(self):
        self.run_pipline_btn = tk.Button(self.root, text="Run Pipeline", command=lambda: self._run_both())
        self.run_pipline_btn.grid(row=2, column=0, padx=10, pady=(0,10), sticky="we")

        self.def_val_btn = tk.Button(self.root, text="Default Values", command=lambda: self.default())
        self.def_val_btn.grid(row=2, column=1, padx=10, pady=(0,10), sticky="we")

        x = tk.Frame(self.root, padx=0, pady=0)
        x.grid(row=3, column=0, columnspan=2, padx=10, pady=0, sticky="nswe")
        x.grid_rowconfigure(0, weight=1)
        x.grid_columnconfigure(0, weight=1)

        self.OUTPUT = tk.Text(x, height=3, state='disabled', wrap='word', font=("Courier", 9))
        self.OUTPUT.grid(row=0, column=0, columnspan=2, padx=0, pady=0, sticky="we")

    ################# VALIDATE FUNCTIONS #################

    def validate_numeric(self, value_if_allowed):
        if value_if_allowed == "" or value_if_allowed.isdigit():
            return True
        else:
            return False

    def validate_numeric_float(self, value_if_allowed):
        if value_if_allowed == "":
            return True
        try:
            float(value_if_allowed)
            return True
        except ValueError:
            return False
    
    def validate_segment(self):
        folder_path = self.folder_path.get()
        if not os.path.isdir(folder_path):
            messagebox.showerror(
                "Error", "Invalid folder. Please select a dataset folder."
            )
            return False

        if self.run_mode.get() == "Single":
            folder_path = self.file_path
            if not os.path.isfile(folder_path):
                messagebox.showerror(
                    "Error", "Invalid file. Please select an image from the dataset."
                )
                return False

        model_path = self.model_path.get()
        if not os.path.isdir(model_path):
            messagebox.showerror(
                "Error", "Invalid path. Please select a valid model path."
            )
            return False

        return True

    def validate_mli(self):
        folder_path = self.folder_path.get()
        if not os.path.isdir(folder_path):
            messagebox.showerror(
                "Error", "Invalid folder. Please select a dataset folder."
            )
            return False

        if self.run_mode.get() == "Single":
            if not os.path.isfile(self.file_path):
                messagebox.showerror(
                    "Error", "Invalid file. Please select an image from the dataset."
                )
                return False
            
            mask_path = self.mask_path.get()
            if not os.path.isdir(mask_path):
                messagebox.showerror(
                    "Error", "Invalid path. Please verify MASKS folder exists."
                )
                return False

        overlap_x_value = int(self.overlap_x.get())
        overlap_y_value = int(self.overlap_y.get())
        if not (0 <= overlap_x_value <= 100) or not (0 <= overlap_y_value <= 100):
            messagebox.showerror(
                "Error", "Overlap percentages must be between 0 and 100% ."
            )
            return False
        
        guidelength = float(self.guidelength.get())
        if not (0 <= guidelength ):
            messagebox.showerror(
                "Error", "Guidelength must be bigger than zero ."
            )
            return False        

        method_dir = self.method_dir.get()
        method_indir = self.method_indir.get()

        if method_dir == "0" and method_indir == "0":
            messagebox.showerror(
                "Error", "At least one calculation method must be selected."
            )
            return False

        return True
    
    ################# ACTION FUNCTIONS #################

    def segment_wsi(self):
        if self.validate_segment():
            self.disable_btns()
            if self.run_mode.get() == "Single":
                try:
                    extract_tiles(
                        file_path=self.file_path,
                        tile_size=int(self.tile_size.get()),
                        overlap_percent=int(self.tile_overlap.get()) / 100,
                    )
                except Exception as e:
                    messagebox.showerror(
                        "Error", f"Error occurred while creating tiles:\n{str(e)}"
                    )
                try:
                    run_eccn(file_path=self.file_path, model_folder=self.model_path.get())
                except Exception as e:
                    messagebox.showerror(
                        "Error",
                        f"Error occurred while the model is predicting first-stage heatmaps:\n{str(e)}",
                    )
                try:
                    reconstruct_heatmaps(file_path=self.file_path)
                except Exception as e:
                    messagebox.showerror(
                        "Error",
                        f"Error occurred while reconstructing first-stage heatmaps:\n{str(e)}",
                    )
                try:
                    extract_hm_tiles(file_path=self.file_path)
                except Exception as e:
                    messagebox.showerror(
                        "Error",
                        f"Error occurred while creating second-stage heatmaps tiles:\n{str(e)}",
                    )
                try:
                    run_accn(file_path=self.file_path, model_folder=self.model_path.get())
                except Exception as e:
                    messagebox.showerror(
                        "Error",
                        f"Error occurred while the model is predicting second-stage heatmaps:\n{str(e)}",
                    )
                try:
                    reconstruct_final_heatmaps(file_path=self.file_path)
                except Exception as e:
                    messagebox.showerror(
                        "Error",
                        f"Error occurred while reconstructing second-stage heatmaps:\n{str(e)}",
                    )
                try:
                    clean_temp(file_path=self.file_path)
                except Exception as e:
                    messagebox.showerror(
                        "Error",
                        f"Error occurred while cleaning temp folders:\n{str(e)}",
                    )

            elif self.run_mode.get() == "Entire":
                for i in self.tif_files:
                    file_path = os.path.join(self.folder_path.get(), i)
                    try:
                        extract_tiles(
                            file_path=file_path,
                            tile_size=int(self.tile_size.get()),
                            overlap_percent=int(self.tile_overlap.get()) / 100,
                        )
                    except Exception as e:
                        messagebox.showerror(
                            "Error", f"Error occurred while creating tiles:\n{str(e)}"
                        )
                    try:
                        run_eccn(file_path=file_path, model_folder=self.model_path.get())
                    except Exception as e:
                        messagebox.showerror(
                            "Error",
                            f"Error occurred while the model is predicting first-stage heatmaps:\n{str(e)}",
                        )
                    try:
                        reconstruct_heatmaps(file_path=file_path)
                    except Exception as e:
                        messagebox.showerror(
                            "Error",
                            f"Error occurred while reconstructing first-stage heatmaps:\n{str(e)}",
                        )
                    try:
                        extract_hm_tiles(file_path=file_path)
                    except Exception as e:
                        messagebox.showerror(
                            "Error",
                            f"Error occurred while creating second-stage heatmaps tiles:\n{str(e)}",
                        )
                    try:
                        run_accn(file_path=file_path, model_folder=self.model_path.get())
                    except Exception as e:
                        messagebox.showerror(
                            "Error",
                            f"Error occurred while the model is predicting second-stage heatmaps:\n{str(e)}",
                        )
                    try:
                        reconstruct_final_heatmaps(file_path=file_path)
                    except Exception as e:
                        messagebox.showerror(
                            "Error",
                            f"Error occurred while reconstructing second-stage heatmaps:\n{str(e)}",
                        )

                    try:
                        clean_temp(file_path=file_path)
                    except Exception as e:
                        messagebox.showerror(
                            "Error",
                            f"Error occurred while cleaning temp folders:\n{str(e)}",
                        )
            self.enable_btns()

    def calc_mli(self):
        guidelength = int(round(2.0084975 *float(self.guidelength.get())))
        if self.run_mode.get() == "Single":
            if self.validate_mli():
                self.disable_btns()
                if self.method_dir.get() == "1" and self.method_indir.get() == "0":
                    try:
                        mli = dir_score(
                            self.mask_path.get(),
                            self.file_path,
                            int(self.fov_x.get()),
                            int(self.fov_y.get()),
                            int(self.overlap_x.get()) / 100,
                            int(self.overlap_y.get()) / 100,
                            guidelength,
                            int(self.save_fov_tf.get()),
                        )
                        self.score_dir.set(f"{mli:.2f}")
                    except Exception as e:
                        messagebox.showerror(
                            "Error",
                            f"Error occurred while calculating direct MLI score:\n{str(e)}",
                        )
                elif self.method_dir.get() == "0" and self.method_indir.get() == "1":
                    try:
                        mli = indir_score(
                            self.mask_path.get(),
                            self.file_path,
                            int(self.fov_x.get()),
                            int(self.fov_y.get()),
                            int(self.overlap_x.get()) / 100,
                            int(self.overlap_y.get()) / 100,
                            guidelength,
                            int(self.save_fov_tf.get()),
                        )
                        self.score_indir.set(f"{mli:.2f}")
                    except Exception as e:
                        messagebox.showerror(
                            "Error",
                            f"Error occurred while calculating indirect MLI score:\n{str(e)}",
                        )
                elif self.method_dir.get() == "1" and self.method_indir.get() == "1":
                    try:
                        mli = dir_score(
                            self.mask_path.get(),
                            self.file_path,
                            int(self.fov_x.get()),
                            int(self.fov_y.get()),
                            int(self.overlap_x.get()) / 100,
                            int(self.overlap_y.get()) / 100,
                            guidelength,
                            int(self.save_fov_tf.get()),
                        )
                        self.score_dir.set(f"{mli:.2f}")
                    except Exception as e:
                        messagebox.showerror(
                            "Error",
                            f"Error occurred while calculating direct MLI score:\n{str(e)}",
                        )
                    try:
                        mli = indir_score(
                            self.mask_path.get(),
                            self.file_path,
                            int(self.fov_x.get()),
                            int(self.fov_y.get()),
                            int(self.overlap_x.get()) / 100,
                            int(self.overlap_y.get()) / 100,
                            guidelength,
                            int(self.save_fov_tf.get()),
                        )
                        self.score_indir.set(f"{mli:.2f}")
                    except Exception as e:
                        messagebox.showerror(
                            "Error",
                            f"Error occurred while calculating indirect MLI score:\n{str(e)}",
                        )
                self.enable_btns()

        elif self.run_mode.get() == "Entire":
            for i in self.tif_files:
                file_path = os.path.join(self.folder_path.get(), i)
                mask_path = os.path.join(self.folder_path.get(), f"{i.replace('.tif', '')}_masks")
                if self.validate_mli():
                    self.disable_btns()
                    if self.method_dir.get() == "1" and self.method_indir.get() == "0":
                        try:
                            mli = dir_score(
                                mask_path,
                                file_path,
                                int(self.fov_x.get()),
                                int(self.fov_y.get()),
                                int(self.overlap_x.get()) / 100,
                                int(self.overlap_y.get()) / 100,
                                guidelength,
                                int(self.save_fov_tf.get()),
                            )
                            self.score_dir.set(f"{mli:.2f}")
                        except Exception as e:
                            messagebox.showerror(
                                "Error",
                                f"Error occurred while calculating direct MLI score:\n{str(e)}",
                            )
                    elif self.method_dir.get() == "0" and self.method_indir.get() == "1":
                        try:
                            mli = indir_score(
                                mask_path,
                                file_path,
                                int(self.fov_x.get()),
                                int(self.fov_y.get()),
                                int(self.overlap_x.get()) / 100,
                                int(self.overlap_y.get()) / 100,
                                guidelength,
                                int(self.save_fov_tf.get()),
                            )
                            self.score_indir.set(f"{mli:.2f}")
                        except Exception as e:
                            messagebox.showerror(
                                "Error",
                                f"Error occurred while calculating indirect MLI score:\n{str(e)}",
                            )
                    elif self.method_dir.get() == "1" and self.method_indir.get() == "1":
                        try:
                            mli = dir_score(
                                mask_path,
                                file_path,
                                int(self.fov_x.get()),
                                int(self.fov_y.get()),
                                int(self.overlap_x.get()) / 100,
                                int(self.overlap_y.get()) / 100,
                                guidelength,
                                int(self.save_fov_tf.get()),
                            )
                            self.score_dir.set(f"{mli:.2f}")
                        except Exception as e:
                            messagebox.showerror(
                                "Error",
                                f"Error occurred while calculating direct MLI score:\n{str(e)}",
                            )
                        try:
                            mli = indir_score(
                                mask_path,
                                file_path,
                                int(self.fov_x.get()),
                                int(self.fov_y.get()),
                                int(self.overlap_x.get()) / 100,
                                int(self.overlap_y.get()) / 100,
                                guidelength,
                                int(self.save_fov_tf.get()),
                            )
                            self.score_indir.set(f"{mli:.2f}")
                        except Exception as e:
                            messagebox.showerror(
                                "Error",
                                f"Error occurred while calculating indirect MLI score:\n{str(e)}",
                            )
                    self.enable_btns()

    def run_both(self):
        self.segment_wsi()
        self.calc_mli()

    def _segment_wsi(self):
        threading.Thread(target=self.segment_wsi, daemon=True).start()

    def _calc_mli(self):
        threading.Thread(target=self.calc_mli, daemon=True).start()

    def _run_both(self):
        threading.Thread(target=self.run_both, daemon=True).start()

    ################# EVENT FUNCTIONS #################

    def select_folder_dataset(self):
        f = filedialog.askopenfilename(initialdir=__file__, filetypes=[("TIF files", "*.tif")], title="Select a TIF file")
        if f:
            self.image_name = os.path.splitext(os.path.basename(f))[0]
            self.folder_path.set(os.path.dirname(f))
            self.file_path = f
            self.file_path_tk.set(f)
            self.mask_path.set(os.path.join(self.folder_path.get(), f"{self.image_name}_masks"))

    def select_folder_model(self):
        x = filedialog.askdirectory(initialdir=__file__)
        if x:
            self.model_path.set(x)

    def collect_tifs(self, event):
        if self.combo.get() == "Entire":
            self.tif_files = []
            if os.path.isdir(self.folder_path.get()):
                for file in os.listdir(self.folder_path.get()):
                    if file.endswith(".tif"):
                        self.tif_files.append(file)

    def default(self):
        self.fov_x.set(1388)
        self.fov_y.set(1072)
        self.overlap_x.set(50)
        self.overlap_y.set(50)
        self.guidelength.set(155.3)
        self.save_fov_tf.set(0)
        self.root.update_idletasks()

    def redirect_logging(self):
        logger = PrintLogger(self.OUTPUT)
        sys.stdout = logger
        sys.stderr = logger

    def disable_btns(self):
        self.select_image_btn.config(state="disable")
        self.select_model_btn.config(state="disable")
        self.create_masks_btn.config(state="disable")
        self.calc_mli_btn.config(state="disable")
        self.run_pipline_btn.config(state="disable")
        self.def_val_btn.config(state="disable")

    def enable_btns(self):
        self.select_image_btn.config(state="normal")
        self.select_model_btn.config(state="normal")
        self.create_masks_btn.config(state="normal")
        self.calc_mli_btn.config(state="normal")
        self.run_pipline_btn.config(state="normal")
        self.def_val_btn.config(state="normal")
    
    ################# CREATE GUI #################

    @classmethod
    def run(cls):
        root = tk.Tk()
        app = cls(root)
        root.mainloop()

if __name__ == "__main__":
    App.run()
