import java.io.*;
import java.util.*;

public class C_NotAssigning_1400 {
    public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));
        int t = sc.nextInt();
        while(t-->0) {
            int n = sc.nextInt();
            boolean[] visited = new boolean[n];
            ArrayList<Edge>[] adj = new ArrayList[n];
            for(int i = 0; i < n; i++) {
                adj[i] = new ArrayList<>();
            }
            for(int i = 0; i < n-1; i++) {
                int a = sc.nextInt()-1;
                int b = sc.nextInt()-1;
                adj[a].add(new Edge(b, i));
                adj[b].add(new Edge(a, i));
            }
            int start = -1;
            boolean flag = false;
            for(int i = 0; i < n; i++) { //start from leaf node
                if(adj[i].size() == 1) {
                    start = i;
                } else if(adj[i].size() > 2) {
                    flag = true;
                }
            }
            int[] weights = new int[n-1];
            Queue<Integer> que = new LinkedList<>();
            que.offer(start);
            visited[start] = true;
            int curDist = 0;
            while(!que.isEmpty()) {
                int cur = que.poll();
                for(Edge e : adj[cur]) {
                    if(!visited[e.to]) {
                        visited[e.to] = true;
                        que.offer(e.to);
                        if(curDist%2 == 0) {
                            weights[e.index] = 2;
                        } else {
                            weights[e.index] = 3;
                        }
                    }
                }
                curDist++;
            }
            if(flag) {
                System.out.println(-1);
            } else {
                StringBuilder sb = new StringBuilder();
                for(int i = 0; i < n-1; i++){
                    sb.append(weights[i] + " ");
                }
                System.out.println(sb.toString().trim());
            }

        }

        out.close();
    }
    static class Edge{
        public int to;
        public int index;
        Edge(int to, int index){
            this.to = to;
            this.index = index;
        }
    }

    public static PrintWriter out;

    

}
