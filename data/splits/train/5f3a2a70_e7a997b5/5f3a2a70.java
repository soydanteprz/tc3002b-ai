import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

public class Main {

    private static Node[] nodes;
    private static boolean[] visited;

    private static void run() throws IOException {
        int n = in.nextInt();

        nodes = new Node[n];
        for (int i = 0; i < n; i++) {
            nodes[i] = new Node();
        }

        Edge[] edges = new Edge[n - 1];

        for (int i = 0; i < n - 1; i++) {
            edges[i] = new Edge();
            edges[i].id = i;
            edges[i].u = in.nextInt() - 1;
            edges[i].v = in.nextInt() - 1;

            nodes[edges[i].u].edge.add(edges[i]);
            nodes[edges[i].v].edge.add(edges[i]);
        }

        for (int i = 0; i < n; i++) {
            if (nodes[i].edge.size() >= 3) {
                out.println(-1);
                return;
            }
        }

        int end_pos = 0;
        visited = new boolean[n];
        while (nodes[end_pos].edge.size() != 1) {
            end_pos++;
        }

        visited = new boolean[n];
        dfs(end_pos, 2);

        for (int i = 0; i < n - 1; i++) {
            out.print(edges[i].ans);
            out.print(' ');
        }
        out.println();
    }

    private static void dfs(int now, int d) {
        visited[now] = true;
        for (Edge edge : nodes[now].edge) {
            if (visited[edge.get_next(now)]) {
                continue;
            }

            edge.ans = d;
            dfs(edge.get_next(now), d ^ 1);
        }
    }

    private static class Node {
        ArrayList<Edge> edge = new ArrayList<>();
    }

    private static class Edge {
        int id, u, v, ans = -1;

        int get_next(int now) {
            return now != u ? u : v;
        }
    }

    public static void main(String[] args) throws IOException {
        in = new Reader();
        out = new PrintWriter(new OutputStreamWriter(System.out));

        int t = in.nextInt();
        for (int i = 0; i < t; i++) {
            run();
        }

        out.flush();
        in.close();
        out.close();
    }

    

    static final long mod = 1000000007;

    

    

    

    @SuppressWarnings("FieldCanBeLocal")
    private static Reader in;
    private static PrintWriter out;

    

    

    

    

    
}
