import java.io.*;
import java.util.*;
import static java.util.Collections.*;
import static java.lang.Math.*;
import java.util.stream.*;

import javax.sound.sampled.Mixer;

import static java.util.Arrays.*;

@SuppressWarnings("unchecked")
public class D_Tree_Tag {
    public static PrintWriter out;
    public static InputReader in;
    public static long MOD = (long)1e9+7;
    public static int n,a,b,da,db;
    public static Graph g;
    public static int dab;
    public static int mxdist;
    static class Graph
    {
        public int n;
        public ArrayList adjacency[];
        public Graph(int nodes)
        {
            n = nodes;
            adjacency = new ArrayList[nodes];
            for(int i=0;i<nodes;i++)
            {
                adjacency[i] = new ArrayList<Integer>();
            }
        }
        
        public void addEdge(int from, int to)
        {
            adjacency[from].add(to);
            adjacency[to].add(from);
        }
    }
    static class Pair{
        public int a,b;
        public Pair(int a, int b){
            this.a=a; this.b=b;
        }
    }
    public static void dist(){
        int[] vis = new int[n]; fill(vis,0);
        LinkedList<Pair> l = new LinkedList<Pair>();
        l.add(new Pair(a,0));
        while(!l.isEmpty()){
            Pair cur = l.pollFirst();
            int v = cur.a, ct = cur.b;
            vis[v] = 1;
            if(v==b) {dab = ct; return;}
            for(int neigh : (ArrayList<Integer>)g.adjacency[v]){
                if(vis[neigh]==0) l.add(new Pair(neigh,ct+1));
            }
        }
    }
    // https://www.geeksforgeeks.org/diameter-tree-using-dfs/#:~:text=We%20start%20DFS%20from%20a,the%20diameter%20of%20the%20tree.
    // Used to track farthest node. 
    static int x; 
    static int maxCount; 
    static List<Integer> adj[]; 
      
    // Sets maxCount as maximum distance  
    // from node 
    static void dfsUtil(int node, int count,  
                         boolean visited[], 
                       List<Integer> adj[]) 
    { 
        visited[node] = true; 
        count++; 
          
        List<Integer> l = adj[node]; 
        for(Integer i: l) 
        { 
            if(!visited[i]){ 
                if (count >= maxCount) { 
                    maxCount = count; 
                    x = i; 
                } 
                dfsUtil(i, count, visited, adj); 
            } 
        } 
    } 
       
    // The function to do DFS traversal. It uses  
    // recursive dfsUtil() 
    static void dfs(int node, int n, List<Integer> 
                                       adj[]) 
    { 
        boolean[] visited = new boolean[n + 1]; 
        int count = 0; 
       
        // Mark all the vertices as not visited 
        Arrays.fill(visited, false); 
       
        // Increment count by 1 for visited node 
        dfsUtil(node, count + 1, visited, adj); 
          
    } 
       
    // Returns diameter of binary tree represented 
    // as adjacency list. 
    static int diameter(List<Integer> adj[], int n) 
    { 
        maxCount = Integer.MIN_VALUE; 
       
        /* DFS from a random node and then see 
        farthest node X from it*/
        dfs(1, n, adj); 
       
        /* DFS from X and check the farthest node 
        from it */
        dfs(x, n, adj); 
       
        return maxCount; 
    } 
    
    public static void main(String[] args)throws IOException {
        in = new InputReader(System.in);
        out = new PrintWriter(System.out);
        
        int cases = in.nextInt();
        
        for(int t = 0; t < cases; t++){
            n = in.nextInt();
            a = in.nextInt()-1;
            b = in.nextInt()-1;
            da = in.nextInt();
            db = in.nextInt();
            g = new Graph(n);
            for(int i=0;i<n-1;i++){
                int u = in.nextInt()-1;
                int v=  in.nextInt()-1;
                g.addEdge(u, v);
            }
            // dab=-1;
            dist();
            mxdist = diameter(g.adjacency, n);
            mxdist--;
            if(dab==-1){
                out.println("FUCK");
            }
            // mxdist = -1;
            // int[] vis = new int[n]; fill(vis,0);
            // mxdist = max(mxdist,dfs2(a,vis));
            // out.println("mxdist="+mxdist);
            if(dab<=da){
                out.println("Alice");
            }
            else{
                if(da>=db) out.println("Alice");
                else{
                    if(db>(2*da) && mxdist>(2*da)){
                        out.println("Bob");
                    }
                    else out.println("Alice");
                }
            }
            // out.println("dab:"+dab);
            // out.println("dist:"+mxdist);
        }

        out.close();
    }
    

}
