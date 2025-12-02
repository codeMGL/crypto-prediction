import matplotlib.pyplot as plt


class Graph:
    def __init__(self):
        # Normalized metrics
        self.mse_norm = {"id": "MSE", "train": [], "test": []}
        self.mae_norm = {"id": "MAE", "train": [], "test": []}
        self.norm_metrics = [self.mse_norm, self.mae_norm]

        # Real metrics (solo MAPE)
        self.mape = {"id": "MAPE", "train": [], "test": []}
        self.real_metrics = [self.mape]

        # Líneas
        self.lines_norm = []
        self.lines_real = []

        # Solo un eje X global
        self.epochs = []

        self.initializeGraph()

    def initializeGraph(self):
        plt.ion()

        self.fig, (self.ax_norm, self.ax_real) = plt.subplots(1, 2, figsize=(14, 6))

        # Normalizados
        self.ax_norm.set_title("Normalized Metrics")
        self.ax_norm.set_xlabel("Epoch")
        self.ax_norm.set_ylabel("Error (Norm)")
        self.ax_norm.grid(True)

        colores = ["blue", "green", "red", "cyan", "magenta"]
        i = 0
        for metric in self.norm_metrics:
            color = colores[i]
            (l_train,) = self.ax_norm.plot(
                [], [], color=color, linestyle="-", label=f"{metric['id']} Train"
            )
            (l_test,) = self.ax_norm.plot(
                [], [], color=color, linestyle="--", label=f"{metric['id']} Test"
            )
            self.lines_norm.append(
                {"line_train": l_train, "line_test": l_test, "data": metric}
            )
            i += 1
        self.ax_norm.legend()

        # Reales (solo MAPE)
        self.ax_real.set_title("Real Value Metrics")
        self.ax_real.set_xlabel("Epoch")
        self.ax_real.set_ylabel("Error (Real)")
        self.ax_real.grid(True)

        for metric in self.real_metrics:
            (l_train,) = self.ax_real.plot(
                [], [], color="cyan", linestyle="-", label=f"{metric['id']} Train"
            )
            (l_test,) = self.ax_real.plot(
                [], [], color="cyan", linestyle="--", label=f"{metric['id']} Test"
            )
            self.lines_real.append(
                {"line_train": l_train, "line_test": l_test, "data": metric}
            )
        self.ax_real.legend()

    def updateAndPlot(
        self,
        mse_norm_train,
        mse_norm_test,
        mae_norm_train,
        mae_norm_test,
        mape_train,
        mape_test,
        epoch=-1,
    ):
        # Añadir epoch una única vez
        if epoch == -1:
            if len(self.epochs) >= 2:
                epoch = self.epochs[-1] - self.epochs[-2]
            elif len(self.epochs) >= 1:
                epoch = self.epochs[-1]
            else:
                epoch = 0
        self.epochs.append(epoch)

        # Añadir datos normalizados
        self.mse_norm["train"].append(mse_norm_train)
        self.mse_norm["test"].append(mse_norm_test)

        self.mae_norm["train"].append(mae_norm_train)
        self.mae_norm["test"].append(mae_norm_test)

        # Añadir datos MAPE reales
        self.mape["train"].append(mape_train)
        self.mape["test"].append(mape_test)

        # Actualizar ambas gráficas
        self.update_lines(self.lines_norm, self.ax_norm)
        self.update_lines(self.lines_real, self.ax_real)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_lines(self, lines_list, ax):
        for item in lines_list:
            datos_train = item["data"]["train"]
            datos_test = item["data"]["test"]

            item["line_train"].set_data(self.epochs, datos_train)
            item["line_test"].set_data(self.epochs, datos_test)

        ax.relim()
        ax.autoscale_view()


# print("NO GRAPH!")
graph = Graph()
