import java.io.*;
import java.util.*;


public class codeforces2 {
    static List<Integer> primes;
    static final long X = 10000000000L;
    public static void main(String[] args) {
        FastScanner sc = new FastScanner();
        PrintWriter pw = new PrintWriter(System.out);
//        primes = sieveOfEratosthenes(100_000);
//        System.out.println(primes.toString());
        int tc = sc.ni();
//        int tc = 1;
        for (int rep = 0; rep < tc; rep++) {
            int N = sc.ni();
            int[][] edges= new int[N-1][];
            for (int i = 0; i < edges.length; i++) {
                edges[i] = new int[] {sc.ni() - 1, sc.ni() - 1};
            }
            pw.println(solve(edges, N));
            

        }
        pw.close();
    }
    public static String solve(int[][] edges, int N) {
        Map<Integer, Set<Integer>> graph = new HashMap();
        Map<Long, Integer> hash = new HashMap();
        int idx = 0;
        for (int[] e : edges) {
            graph.putIfAbsent(e[0], new HashSet());
            graph.get(e[0]).add(e[1]);
            graph.putIfAbsent(e[1], new HashSet());
            graph.get(e[1]).add(e[0]);
            
            
            
            hash.put(getHash(e[0], e[1]), idx);
            idx++;
        }
        
        Queue<int[]> q = new LinkedList();
        //node, previous weight
        int[] initial = new int[2];
        for (int node : graph.keySet()) {
            int len = graph.get(node).size();
            if (len > 2) return "-1";
            if (len == 2) initial = new int[] {node, -1};
        }
        q.offer(initial);
        
        int rotate = 0;
//        System.out.println(graph.toString());
           
        int[] ret = new int[edges.length];
        boolean[] vis = new boolean[N];
        vis[initial[0]] = true;
        while (!q.isEmpty()) {
            int[] temp = q.poll();
            int node = temp[0];
            int old_color = temp[1];
            
            if (old_color == -1) {
                int[] arr = new int[] {2, 11};
                int p = 0;
                for (int neighbor : graph.get(node)) {
                    ret[hash.get( getHash(node,neighbor) )] = arr[p];
                    q.offer(new int[] {neighbor, arr[p]});
                    p++;
//                    System.out.println();
//                    System.out.println(neighbor + " " + graph.toString());
//                    System.out.println(Arrays.deepToString(edges));
                    vis[neighbor] = true;
                    
                }
                
            }
            else if (old_color == 2) {
                
                int amt = 0;
                for (int neighbor : graph.get(node)) {
                    if (vis[neighbor] == true) continue;
                    vis[neighbor] = true;
                    
                    ret[hash.get( getHash(node,neighbor) )] = 11;
                    q.offer(new int[] {neighbor, 11});
                    
                    amt++;
                }
                if (amt >= 2) return "-1";
            }
            else {
                
                int amt = 0;
                for (int neighbor : graph.get(node)) {
                    if (vis[neighbor] == true) continue;
                    vis[neighbor] = true;
                    
                    ret[hash.get( getHash(node,neighbor) )] = 2;
                    q.offer(new int[] {neighbor, 2});
                    
                    amt++;
                }
                if (amt >= 2) return "-1";
            
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int x : ret) sb.append(x + " ");
        return sb.substring(0, sb.length() -1);
        
    }
    static long getHash(int u, int v) {
        
        if (u > v) {
            int temp = u;
            u = v;
            v = temp;
        }
        return u*X+v;
    }
        
     
   
    
    
    
    
    
    
    
    
    
     
    // method to return LCM of two numbers
    
    
 
    
    /* */
    //printing methods
    /* */
    //WOW!
    /* */
    
    
}




