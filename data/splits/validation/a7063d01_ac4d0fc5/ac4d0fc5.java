import java.io.*;
import java.util.*;

public class Main {
    static boolean[] ret;
    static boolean[] updated;
    static ArrayList<Integer>[] adjacencyList;
    static Edge[] edgeList;
    static class Edge {
        int start, end, number;
        public Edge (int _start, int _end, int _number) {
            start = _start;
            end = _end;
            number = _number;
        }
    }
    public static void dfs(int node) {
        updated[node] = true;
        for (int next : adjacencyList[edgeList[node].start]) {
            if (!updated[next]) {
                ret[next] = !ret[node];
                dfs(next);
            }
        }
        for (int next : adjacencyList[edgeList[node].end]) {
            if (!updated[next]) {
                ret[next] = !ret[node];
                dfs(next);
            }
        }
    }
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
        int numCases = Integer.parseInt(br.readLine());
        for (int i = 0; i < numCases; i++) {
            int numVertices = Integer.parseInt(br.readLine());
            int[] numEdges = new int[numVertices];
            edgeList = new Edge[numVertices - 1];
            adjacencyList = new ArrayList[numVertices];
            for (int j = 0; j < numVertices; j++) {
                adjacencyList[j] = new ArrayList<>();
            }
            for (int j = 0; j < numVertices - 1; j++) {
                StringTokenizer st = new StringTokenizer(br.readLine());
                int a = Integer.parseInt(st.nextToken()) - 1;
                int b = Integer.parseInt(st.nextToken()) - 1;
                edgeList[j] = new Edge(a, b, j);
                numEdges[a]++;
                numEdges[b]++;
                adjacencyList[a].add(j);
                adjacencyList[b].add(j);
            }
            boolean good = true;
            for (int j = 0; j < numVertices; j++) {
                if (numEdges[j] > 2) {
                    good = false;
                    break;
                }
            }
            if (!good) {
                pw.println(-1);
            } else {
                ret = new boolean[numVertices - 1];
                updated = new boolean[numVertices - 1];
                dfs(0);
                for (boolean b : ret) {
                    if (b)
                        pw.print(5 + " ");
                    else
                        pw.print(2 + " ");
                }
                pw.println();
            }
        }
        br.close();
        pw.close();
    }
}
