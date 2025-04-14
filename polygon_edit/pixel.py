import sys
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QRectF

class PixelViewer(QGraphicsView):
    def __init__(self, image_path):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Load the image
        self.image = QImage(image_path)
        self.pixmap = QPixmap.fromImage(self.image)
        self.scene.addPixmap(self.pixmap)

        # Draw the grid
        self.draw_grid()

        # Enable mouse tracking
        self.setMouseTracking(True)

    def draw_grid(self):
        for x in range(self.image.width()):
            for y in range(self.image.height()):
                rect = QRectF(x, y, 1, 1)
                grid_item = QGraphicsRectItem(rect)
                grid_item.setPen(QColor(200, 200, 200, 0)) 
                self.scene.addItem(grid_item)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            if 0 <= x < self.image.width() and 0 <= y < self.image.height():
                print(f"Clicked on pixel: x={x}, y={y}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    image_path = "/home/barun/mundi/kcv/yolov8-streamlit-detection-tracking/polygon_edit/put_pixel_2.jpg"
    viewer = PixelViewer(image_path)
    viewer.setWindowTitle("Pixel Viewer")
    viewer.resize(1000, 800)
    viewer.show()
    sys.exit(app.exec_())