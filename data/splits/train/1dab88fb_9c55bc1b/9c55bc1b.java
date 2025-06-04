import java.util.*;
import java.io.*;

public class Main {
    static MyScanner sc;
    static PrintWriter out;
    static {
        sc = new MyScanner();
        out = new PrintWriter(System.out);
    }
    public static void bfs(Node[] g, int[] ans) {
        Arrays.fill(ans, -1);
        boolean[] visited = new boolean[g.length];
        Queue<Integer> q = new LinkedList<>();
        int s = 0;
        for(int i = 0; i < g.length; i++) {
            if(g[i].l.size() == 1) {
                s = i;
                break;
            }
        }
        q.add(s);
        int curr = 2;
        while(!q.isEmpty()) {
            int u = q.poll();
            if(visited[u])
                continue;
            visited[u] = true;
            for(Edge edge : g[u].l) {
                if(!visited[edge.v]) {
                    ans[edge.id] = curr;
                    q.add(edge.v);
                    if(curr == 2) curr = 3;
                    else curr = 2;
                }
            }
        }
    }
    public static void solve() {
        int n = sc.nextInt();
        Node[] a = new Node[n];
        for(int i = 0; i < n; i++)
            a[i] = new Node();
        for(int i = 0; i < n - 1; i++) {
            int u = sc.nextInt() - 1;
            int v = sc.nextInt() - 1;
            a[u].l.add(new Edge(v, i));
            a[v].l.add(new Edge(u, i));
        }
        for(Node node : a) {
            if(node.l.size() > 2) {
                out.println(-1);
                return;
            }
        }
        int[] ans = new int[n - 1];
        bfs(a, ans);
        for(int i = 0; i < n - 1; i++)
            out.print(ans[i] + " ");
        out.println();
    }
    public static void main(String[] args) {
        int t = sc.nextInt();
        while(t-- > 0)
            solve();
        out.flush();
    }
}

class Edge {
    int v, id;
    Edge(int a, int b) {
        v = a;
        id = b;
    }
}

class Node {
    ArrayList<Edge> l;
    Node() {
        l = new ArrayList<>();
    }
}


