import java.io.*;
import java.util.*;

public class c {

    public static void main(String[] args){

        FastScanner sc = new FastScanner();
        int t = sc.nextInt();
        while(t-- > 0){

            int n = sc.nextInt();


            ArrayList<ArrayList<Edge>> graph = new ArrayList<>();
            for(int i=0; i<n; i++){
                graph.add(new ArrayList<Edge>());
            }

            for(int i=0; i<n-1; i++){
                int u = sc.nextInt();
                int v = sc.nextInt();

                Edge e = new Edge(u-1, v-1, i+1);
                Edge e2 = new Edge(v-1, u-1, i+1);

                graph.get(u-1). add(e);
                graph.get(v-1).add(e2);

            }

            int edges[] = new int[n];

            int indegree1count = 0;
            int indegree2count = 0;
            for(ArrayList<Edge> list : graph){
                if(list.size() == 1){
                    indegree1count++;
                }
                else if(list.size() == 2){
                    indegree2count++;
                }
            }

            if(indegree1count == 2 && indegree1count+indegree2count==n){

                for(int i=0; i<graph.size(); i++){
                    ArrayList<Edge> list = graph.get(i);
                    if(list.size() == 1){
                        dfs(graph, edges, false, -1, i) ;
                    }
                }

                for(int i=1; i<edges.length; i++){
                    System.out.print(edges[i] + " ");
                }
                System.out.println();

            }
            else{
                System.out.println(-1);
            }


        }

    }

    public static void dfs(ArrayList<ArrayList<Edge>> graph, int[] edges, boolean isprev2, int parent, int current){

        for(Edge e : graph.get(current)){

            if(e.v == parent){
                continue;
            }

            edges[e.id] = isprev2 ? 5 : 2;

            dfs(graph, edges, !isprev2, current, e.v);

        }

    }

}

class Edge {

    int u;
    int v;
    int id;

    public Edge(int u, int v, int id) {
        this.u = u;
        this.v = v;
        this.id = id;
    }

}


