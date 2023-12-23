import streamlit as st
import numpy as np
import heapq  # Add this line
import matplotlib.pyplot as plt
from io import BytesIO

class SearchAlgorithmApp:
    def __init__(self):
        self.box_size = 20
        self.obstacle_density = 0.2
        self.path = []

    def get_neighbors(self, current):
        x, y = current
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(x, y) for (x, y) in neighbors if 0 <= x < self.box_size and 0 <= y < self.box_size and (x, y) not in self.get_obstacles()]

    def get_obstacles(self):
        predefined_obstacles = [(1, 2), (0, 2), (3, 2), (4, 2), (3, 2), (6, 2), (7, 2),
                                (7, 3), (0, 4), (7, 5), (7, 6), (0, 6), (5, 9), (4, 6),
                                (3, 6), (2, 6), (2, 8), (2, 4), (2, 3), (0,3), (12,9), 
                                (2,17),(0,16),(1,19),(3,17),(12,19),(5,0),(7,1),
                                (14,11),(11,14),(12,12),(5,17),(6,15),(9,11),
                                (15,8), (16, 6), (12,18)]
        return predefined_obstacles

    def run_algorithm(self, algorithm):
        start, end = (0, 0), (self.box_size - 1, self.box_size - 1)

        if algorithm == "Dijkstra":
            iterations, progress = self.dijkstra_algorithm(start, end)
        elif algorithm == "A*":
            iterations, progress = self.a_star_algorithm(start, end)
        elif algorithm == "BFS":
            iterations, progress = self.bfs_algorithm(start, end)
        elif algorithm == "DFS":
            iterations, progress = self.dfs_algorithm(start, end)
        else:
            st.error("Invalid algorithm selected.")
            return

        self.show_map()
        self.plot_graph(iterations, progress)

    def dijkstra_algorithm(self, start, end):
        visited = set()
        heap = [(0, start, [])]  # Updated heap to include the path taken
        iterations = []
        progress = []

        while heap:
            (cost, current, path) = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)
            iterations.append(len(visited))
            progress.append(cost)

            if current == end:
                self.path = path + [current]  # Store the final path
                return iterations, progress

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in self.get_obstacles():
                    heapq.heappush(heap, (cost + 1, neighbor, path + [current]))

        return iterations, progress

    def a_star_algorithm(self, start, end):
        open_set = {start}
        closed_set = set()
        g_scores = {start: 0}
        f_scores = {start: self.distance(start, end)}
        iterations = []
        progress = []

        while open_set:
            current = min(open_set, key=lambda node: f_scores[node])

            if current == end:
                path = self.reconstruct_path(start, end, g_scores.copy())  # Passed a copy of g_scores
                self.path = path
                return iterations, progress

            open_set.remove(current)
            closed_set.add(current)

            iterations.append(len(closed_set))
            progress.append(g_scores.get(current, 0))  # Use get to avoid KeyError

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor in closed_set or neighbor in self.get_obstacles():
                    continue

                tentative_g_score = g_scores.get(current, 0) + 1  # Use get to avoid KeyError

                if neighbor not in open_set or tentative_g_score < g_scores.get(neighbor, 0):  # Use get to avoid KeyError
                    open_set.add(neighbor)
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.distance(neighbor, end)

        return iterations, progress

    def bfs_algorithm(self, start, end):
        queue = [(0, start, [])]  # Updated queue to include the path taken
        visited = set()
        iterations = []
        progress = []

        while queue:
            (cost, current, path) = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)
            iterations.append(len(visited))
            progress.append(cost)

            if current == end:
                self.path = path + [current]  # Store the final path
                return iterations, progress

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in self.get_obstacles():
                    queue.append((cost + 1, neighbor, path + [current]))

        return iterations, progress

    def dfs_algorithm(self, start, end):
        stack = [(0, start, [])]  # Updated stack to include the path taken
        visited = set()
        iterations = []
        progress = []

        while stack:
            (cost, current, path) = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            iterations.append(len(visited))
            progress.append(cost)

            if current == end:
                self.path = path + [current]  # Store the final path
                return iterations, progress

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited and neighbor not in self.get_obstacles():
                    stack.append((cost + 1, neighbor, path + [current]))

        return iterations, progress

    def reconstruct_path(self, start, end, g_scores, max_iterations=1000):
        path = set()
        current = end
        iterations = 0

        while current != start and iterations < max_iterations:
            path.add(current)
            neighbors = self.get_neighbors(current)

            # Find the unvisited neighbor with the lowest cost (g_score)
            current = min(neighbors, key=lambda n: g_scores.get(n, float('inf')))
            iterations += 1

        path.add(start)
        return path

    def distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plot_graph(self, iterations, progress):
        fig, ax = plt.subplots()
        ax.plot(iterations, progress, label='Progress')
        ax.legend()
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Progress (Cost)')

        # Use st.pyplot to display the figure
        st.pyplot(fig)

    def show_map(self):
        # Display the map view above the other graphs
        st.image(self.create_map_image(), use_column_width=True, caption='Map View')

    def create_map_image(self):
        fig, ax = plt.subplots()
        ax.add_patch(plt.Rectangle((0, 0), self.box_size, self.box_size, fill=None))
        
        # Draw obstacles
        for obstacle in self.get_obstacles():
            x, y = obstacle
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color='gray'))

        # Draw path
        for node in self.path:
            x, y = node
            ax.add_patch(plt.Circle((x + 0.5, y + 0.5), 0.4, color='green'))

        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        ax.set_aspect('equal', adjustable='datalim')

        # Save the figure to a BytesIO buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)  # Close the original figure to avoid double plotting

        return buffer.getvalue()

def main():
    st.title("Search Algorithm GUI")

    app = SearchAlgorithmApp()

    # Move the algorithm selection to the main screen
    algorithm = st.selectbox("Select Algorithm", ["Dijkstra", "A*", "BFS", "DFS"])
    if st.button("Run Algorithm"):
        app.run_algorithm(algorithm)

if __name__ == "__main__":
    main()
