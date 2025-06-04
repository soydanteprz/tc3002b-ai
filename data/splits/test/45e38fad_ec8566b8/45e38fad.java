import java.util.*;
import java.io.*;



public class C_Not_Assigning {
    // For fast input output
    

    static class Edge{
        public int node;
        public int next;
        Edge(int node, int next){
            this.node = node;
            this.next = next;
        }    
    }
    public static void dfs(int u, ArrayList<ArrayList<Edge>> g,int[] vis, int flag, int[] weight){
        vis[u] = 1;
        for(Edge e : g.get(u)){
            if(vis[e.node] == 0){
                if(flag == 0)
                {
                    weight[e.next] = 2;
                    dfs(e.node, g, vis, 1, weight);
                }
                else
                {
                    weight[e.next] = 5;
                    dfs(e.node, g, vis, 0, weight);
                }
            }
        }
    }
    // end of fast i/o code
    public static void main(String[] args) {
        FastReader reader = new FastReader();
        int t = reader.nextInt();
        while(t-->0){
            int n = reader.nextInt();
            int[] order = new int[n+1];
            int flag = 0; 
            //ArrayList<ArrayList<Integer>> adj = new ArrayList<>(n+1);
            //Map<Integer, List<Integer>> adj = new HashMap<>();
            ArrayList<ArrayList<Edge>> graph= new ArrayList<>();
            for(int i=0;i<n+1;i++){
                graph.add(new ArrayList<>());
            }
            for(int i=0;i<n-1;i++){
                int u = reader.nextInt();
                int v = reader.nextInt();
                order[u]++;
                order[v]++;
                graph.get(u).add(new Edge(v, i));
                graph.get(v).add(new Edge(u, i));

                if(order[u]==3 || order[v]==3){
                    flag = 1;
                }
            }
            if(flag == 1)
                System.out.println(-1);
            else{
                for(int i=1;i<=n;i++){
                    if(order[i] == 1){
                        int[] vis = new int[n+1];
                        int[] weight = new int[n];
                        dfs(i, graph, vis, 0, weight);
                        for(int j =0;j<n-1;j++){
                            System.out.print(weight[j] + " ");
                        }
                        break;
                    }
                }
                System.out.println();
            }
        }
    }
}
