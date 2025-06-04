import java.io.*; 
import java.util.*; 

public class D  
{
   public static ArrayList<Integer> adj[];
   public static int node;
   public static int dist = 0;
   public static void main(String[] args) throws IOException
   {
      FastScanner sc = new FastScanner();
      int T = sc.nextInt();
      PrintWriter out = new PrintWriter(System.out);
      for(int t = 0; t < T; t++){
         int N = sc.nextInt();
         int a = sc.nextInt(); int b = sc.nextInt();
         int da = sc.nextInt(); int db = sc.nextInt();
         adj = new ArrayList[N+1];
         for(int i = 0; i <= N; i++){
            adj[i] = new ArrayList<Integer>();
         }
         for(int i = 0; i < N-1; i++){
            int v = sc.nextInt();
            int u = sc.nextInt();
            adj[v].add(u);
            adj[u].add(v);
         }
         if(db > 2*da){
            dfs1(a, 0, b, 0);
            if(dist <= da){
               out.println("Alice");
            }
            else{
               node = 0;
               dist = 0;
               dfs(1, 0, 0);
               dfs(node, 0, 0);
               if(dist > 2*da){
                  out.println("Bob");
               }
               else{
                  out.println("Alice");
               }
            }
         }
         else{
            out.println("Alice");
         }
      }
      out.close();
   }  
   public static void dfs1(int a, int p, int b, int d){
      if(a == b){
         dist = d;
      }
      for(int next : adj[a]){
         if(next != p){
            dfs1(next, a, b, d+1);
         }  
      }
   }
   public static void dfs(int i, int p, int d){
      if(d > dist){
         node = i;
         dist = d;
      }
      for(int next : adj[i]){
         if(next != p){
            dfs(next, i, d+1);
         }  
      }
   }


   

   
}
