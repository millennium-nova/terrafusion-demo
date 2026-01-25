import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QPoint


class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(800, 800)  # キャンバスのサイズ
        self.canvas = QImage(self.size(), QImage.Format.Format_RGB32)
        self.canvas.fill(Qt.GlobalColor.black)  # キャンバスの背景を黒に設定

        self.history = []  # 描画履歴
        self.save_state()  # 初期状態を保存

        self.drawing = False
        self.last_point = QPoint()
        self.current_color = QColor(255, 0, 0)  # デフォルト色（赤）
        self.pen_width = 2

    def set_color(self, color):
        """描画色を変更"""
        self.current_color = color

    def save_state(self):
        """現在のキャンバス状態を履歴に保存"""
        self.history.append(self.canvas.copy())

    def undo(self):
        """1つ前の状態に戻る"""
        if len(self.history) > 1:
            self.history.pop()  # 最新の状態を削除
            self.canvas = self.history[-1]  # 1つ前の状態を復元
            self.update()  # 再描画

    def reset_canvas(self):
        """キャンバスをリセット"""
        self.canvas.fill(Qt.GlobalColor.black)  # 背景を黒で塗りつぶす
        self.history = []  # 履歴をリセット
        self.save_state()  # 初期状態を保存
        self.update()  # 再描画

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()
            self.save_state()  # 描画前の状態を保存

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.canvas)
            pen = QPen(self.current_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            painter.end()
            self.update()  # 再描画

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        """描画イベント"""
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.canvas, self.canvas.rect())

    def save_image(self, file_path="sketch_result.png"):
        """描画結果を保存"""
        if isinstance(file_path, str):
            try:
                self.canvas.save(file_path, "PNG")
                print(f"画像を保存しました: {file_path}")
            except Exception as e:
                print(f"保存中にエラーが発生しました: {e}")
        else:
            print("エラー: file_path は文字列である必要があります。")


class SketchTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Terrain Sketch Tool")
        self.setGeometry(100, 100, 1000, 800)  # 幅を少し広げてボタンを配置

        self.canvas = Canvas(self)

        # ボタンの設定
        red_button = QPushButton("谷（赤線）")
        green_button = QPushButton("尾根（緑線）")
        blue_button = QPushButton("崖（青線）")
        undo_button = QPushButton("戻る")
        reset_button = QPushButton("リセット")  # 新しいリセットボタン
        save_button = QPushButton("保存")

        # ボタンのアクション設定
        red_button.clicked.connect(lambda: self.canvas.set_color(QColor(255, 0, 0)))  # 赤
        green_button.clicked.connect(lambda: self.canvas.set_color(QColor(0, 255, 0)))  # 緑
        blue_button.clicked.connect(lambda: self.canvas.set_color(QColor(0, 0, 255)))  # 青
        undo_button.clicked.connect(self.canvas.undo)  # Undo
        reset_button.clicked.connect(self.canvas.reset_canvas)  # リセット
        save_button.clicked.connect(lambda: self.canvas.save_image("sketch_result.png"))

        # レイアウトの設定
        button_layout = QVBoxLayout()  # ボタンを縦に並べる
        button_layout.addWidget(red_button)
        button_layout.addWidget(green_button)
        button_layout.addWidget(blue_button)
        button_layout.addWidget(undo_button)
        button_layout.addWidget(reset_button)  # リセットボタンを追加
        button_layout.addWidget(save_button)
        button_layout.addStretch()  # 空白を追加してボタンが中央寄せになるようにする

        main_layout = QHBoxLayout()  # メインレイアウト（横配置）
        main_layout.addWidget(self.canvas)  # キャンバスを左側に
        main_layout.addLayout(button_layout)  # ボタンを右側に

        # メインウィジェットにレイアウトを設定
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SketchTool()
    window.show()
    sys.exit(app.exec())
