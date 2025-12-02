import matplotlib.pyplot as plt


class Graph:
    def __init__(self):
        # --- 1. Definición de Datos ---
        # Normalized metrics
        self.mse_norm = {"id": "MSE", "train": [], "test": []}
        self.mae_norm = {"id": "MAE", "train": [], "test": []}
        # Lista de diccionarios para iterar fácilmente
        self.norm_metrics = [self.mse_norm, self.mae_norm]

        # Real metrics
        self.rmse = {"id": "RMSE", "train": [], "test": []}
        self.mae = {"id": "MAE", "train": [], "test": []}
        self.mape = {"id": "MAPE", "train": [], "test": []}
        self.real_metrics = [self.rmse, self.mae, self.mape]

        # Listas para guardar las referencias a las líneas de dibujo (Objetos Line2D)
        # Esto nos permitirá actualizarlas luego sin borrarlas
        self.lines_norm = []
        self.lines_real = []

        # Guarda el valor del eje X (lista de enteros que representan los 'epochs')
        self.epochs = []

        # --- 2. Inicialización Gráfica ---
        self.initializeGraph()

    def initializeGraph(self):
        plt.ion()  # Modo interactivo

        # Creamos una figura con 2 columnas (1 fila, 2 columnas)
        # ax_norm será el gráfico izquierdo, ax_real el derecho
        self.fig, (self.ax_norm, self.ax_real) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Configuración Gráfico 1 (Normalizados) ---
        self.ax_norm.set_title("Normalized Metrics")
        self.ax_norm.set_xlabel("Epoch")
        self.ax_norm.set_ylabel("Error (Norm)")
        self.ax_norm.grid(True)

        # Crear líneas iniciales para Normalizados
        # Usamos un índice 'i' manual para los colores en lugar de enumerate
        colores = ["blue", "green", "red", "cyan", "magenta"]
        i = 0
        for metric in self.norm_metrics:
            color = colores[i]
            # Línea de Train (Sólida)
            (l_train,) = self.ax_norm.plot(
                [], [], color=color, linestyle="-", label=f"{metric['id']} Train"
            )
            # Línea de Test (Discontinua)
            (l_test,) = self.ax_norm.plot(
                [], [], color=color, linestyle="--", label=f"{metric['id']} Test"
            )

            # Guardamos las líneas y la referencia a los datos en una lista
            self.lines_norm.append(
                {"line_train": l_train, "line_test": l_test, "data": metric}
            )
            i += 1
        self.ax_norm.legend()

        # --- Configuración Gráfico 2 (Reales) ---
        self.ax_real.set_title("Real Value Metrics")
        self.ax_real.set_xlabel("Epoch")
        self.ax_real.set_ylabel("Error (Real)")
        self.ax_real.grid(True)

        # Crear líneas iniciales para Reales
        for metric in self.real_metrics:
            (l_train,) = self.ax_real.plot(
                [], [], color='cyan', linestyle="-", label=f"{metric['id']} Train"
            )
            (l_test,) = self.ax_real.plot(
                [], [], color='cyan', linestyle="--", label=f"{metric['id']} Test"
            )

            self.lines_real.append(
                {"line_train": l_train, "line_test": l_test, "data": metric}
            )
            i += 1
        self.ax_real.legend()

    def updateAndPlot(
        self,
        mse_norm_train,
        mse_norm_test,
        rmse_real_train,
        rmse_real_test,
        mae_norm_train,
        mae_norm_test,
        mape_train,
        mape_test,
        epoch=0,
    ):
        # 1. Añadir los nuevos datos a las listas
        self.mse_norm["train"].append(mse_norm_train)
        self.mse_norm["test"].append(mse_norm_test)

        self.mae_norm["train"].append(mae_norm_train)
        self.mae_norm["test"].append(mae_norm_test)

        self.rmse["train"].append(rmse_real_train)
        self.rmse["test"].append(rmse_real_test)

        self.mape["train"].append(mape_train)
        self.mape["test"].append(mape_test)

        
        self.epochs.append(epoch)
        # 2. Llamar a la función de dibujo para cada gráfico
        self.update_lines(self.lines_norm, self.ax_norm)
        self.update_lines(self.lines_real, self.ax_real)

        # 3. Refrescar la ventana una sola vez al final
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_lines(self, lines_list, ax):
        # Iteramos sobre las líneas guardadas
        for item in lines_list:
            # Obtenemos los datos actuales
            datos_train = item["data"]["train"]
            datos_test = item["data"]["test"]

            # Actualizamos los datos de las líneas
            item["line_train"].set_data(self.epochs, datos_train)
            item["line_test"].set_data(self.epochs, datos_test)

        # Importante: Re-calcular los límites de los ejes porque los datos han cambiado
        ax.relim()
        ax.autoscale_view()


graph = Graph()
