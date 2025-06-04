import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.LinkedList;

public class C {

    private static boolean[] visited;
    private static Node[] nodes;

    private static void run() throws IOException {
        int n = in.nextInt();
        nodes = new Node[n];
        for (int i = 0; i < n; i++) {
            nodes[i] = new Node();
            nodes[i].index = i;
            nodes[i].l = in.nextInt();
            nodes[i].r = in.nextInt();
        }
        for (int i = 0; i < n - 1; i++) {
            int x = in.nextInt() - 1;
            int y = in.nextInt() - 1;
            nodes[x].next.add(y);
            nodes[y].next.add(x);
        }

        visited = new boolean[n];
        dfs(0);
        out.println(Math.max(nodes[0].ans_l, nodes[0].ans_r));
    }

    private static void dfs(int now) {
        visited[now] = true;

        for (int next : nodes[now].next) {
            if (visited[next]) continue;

            dfs(next);
            nodes[now].ans_l += Math.max(nodes[next].ans_l + Math.abs(nodes[now].l - nodes[next].l), nodes[next].ans_r + Math.abs(nodes[now].l - nodes[next].r));
            nodes[now].ans_r += Math.max(nodes[next].ans_l + Math.abs(nodes[now].r - nodes[next].l), nodes[next].ans_r + Math.abs(nodes[now].r - nodes[next].r));
        }
    }

    static class Node {
        int index;
        long l, r;
        long ans_l, ans_r;
        LinkedList<Integer> next = new LinkedList<>();
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
