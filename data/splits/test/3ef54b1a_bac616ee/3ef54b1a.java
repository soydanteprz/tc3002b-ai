import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        FastIO fio = new FastIO();

        int t = fio.nextInt();

        for (int i = 0; i < t; i++) {
            int n = fio.nextInt();
            ArrayList<ArrayList<Tuple>> adjList = new ArrayList<>();

            for (int j = 0; j < n; j++) {
                adjList.add(new ArrayList<>());
            }

            for (int j = 0; j < n - 1; j++) {
                int u = fio.nextInt() - 1;
                int v = fio.nextInt() - 1;
                adjList.get(u).add(new Tuple(v, j));
                adjList.get(v).add(new Tuple(u, j));
            }

            boolean possible = true;

            for (ArrayList<Tuple> neighbours : adjList) {
                if (neighbours.size() > 2) {
                    possible = false;
                    break;
                }
            }

            if (!possible) {
                fio.println(-1);
                continue;
            }

            int[] assignments = new int[n - 1];
            boolean[] visited = new boolean[n];
            visited[0] = true;
            Queue<Integer> queue = new LinkedList<>();
            queue.offer(0);

            while (!queue.isEmpty()) {
                int u = queue.poll();
                List<Tuple> neighbors = adjList.get(u);

                for (int j = 0; j < neighbors.size(); j++) {
                    Tuple tt = neighbors.get(j);
                    int pi = j;

                    if (neighbors.size() == 2 && visited[neighbors.get(1 - j).v]) {
                        pi = 1 - assignments[neighbors.get(1 - j).num];
                    }

                    if (!visited[tt.v]) {
                        assignments[tt.num] = pi;
                        visited[tt.v] = true;
                        queue.offer(tt.v);
                    }
                }
            }

            for (int j = 0; j < n - 1; j++) {
                if (j > 0) {
                    fio.print(" ");
                }

                fio.print(assignments[j] == 0 ? 2 : 3);
            }

            fio.println();
        }

        fio.close();
    }
}

class Tuple {
    int v, num;

    Tuple(int v, int num) {
        this.v = v;
        this.num = num;
    }
}



/**
 * Fast I/O
 * @source https://www.geeksforgeeks.org/fast-io-in-java-in-competitive-programming/
 */

