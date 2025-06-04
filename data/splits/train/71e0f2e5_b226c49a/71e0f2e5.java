import java.util.*;
import java.lang.*;
import java.io.*;



public class Main {
    
   
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int T = sc.nextInt();
        for(int t=0;t<T;t++){
            solve(sc);       
        }      
    }
    
    static class Edge{
        public int node;
        public int index;
        Edge(int node, int index){
            this.node = node;
            this.index = index;
        }
    }
    public static void solve(Scanner sc){
        int V = sc.nextInt();
        Boolean[] visited = new Boolean[V];
        LinkedList<Edge>[] adj = new LinkedList[V];
       
        for(int v=0;v<V;v++){
            visited[v] = false;
        }
        
        
        for(int v=0;v<V;v++){
            adj[v] = new LinkedList();
        }
        
        for(int i=0;i<V-1;i++){
            int u = sc.nextInt()-1;
            int v = sc.nextInt()-1;
            adj[u].add(new Edge(v,i));
             adj[v].add(new Edge(u,i));
        }
        
       
        //gotta go find the end or start
        int start = -1;
        for(int v=0;v<V;v++){
            if(adj[v].size() == 1){
                start = v;
            }else if(adj[v].size() > 2){
                System.out.println("-1");
                return;
            }
        }
        
        int[] weight = new int[V-1];
       
        //
        int curNode = start;
        int prevNode = -1;
        
        
        visited[start] = true; 
        
        for(int v=0;v<V;v++){
            for(Edge e : adj[start]){
                if(!visited[e.node]){
                    if(v % 2 == 0){
                       weight[e.index] = 2;
                    }else{
                       
                        weight[e.index] = 5;
                    }
                    visited[e.node] = true;
                    start = e.node;
                }  
            }
            
            
        }
        
        for(int w : weight){
            System.out.print(w + " ");
        }
        System.out.println("");
        
        
        
        
    }
}
