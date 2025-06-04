import java.lang.reflect.Array;
import java.util.*;
import java.io.*;
import java.util.regex.Pattern;

public class B2 {

    
    

    
    public static long[] maxBeauty(HashMap[] arr, long[] lv,long[] rv,int start,int parent,boolean[] vis){
        vis[start] = true;
        long[] ans  = new long[2];
       // System.out.println(start);
        ArrayList<Integer> list = new ArrayList<Integer>(arr[start].keySet());
        for(int node:list){
            if(vis[node]==false) {
                long[] curr = maxBeauty(arr,lv,rv,node,start,vis);
                ans[0] += Math.max(curr[0] + Math.abs(lv[start]-lv[node]),curr[1] + Math.abs(lv[start]-rv[node]));
                ans[1]  += Math.max(curr[0] + Math.abs(rv[start]-lv[node]),curr[1] + Math.abs(rv[start]-rv[node]));
            }
        }
        return ans;
    }

    public static void main(String args[]) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter pw = new PrintWriter(System.out);
        int tc = Integer.parseInt(br.readLine());
        while (tc > 0) {
            int n = Integer.parseInt(br.readLine());
            boolean[] vis = new boolean[n+1];
            HashMap[] arr = new HashMap[n+1];
            for(int i=0;i<=n;i++){
                arr[i] = new HashMap<Integer,Long>();
            }
            long[] lv = new long[n+1];
            long[] rv = new long[n+1];
            for(int i=1;i<=n;i++){
                String[] s1 = br.readLine().split(" ");
                lv[i] = Long.valueOf(s1[0]);
                rv[i] = Long.valueOf(s1[1]);
            }
            for(int i=0;i<n-1;i++){
                String[] s1 = br.readLine().split(" ");
                int u = Integer.valueOf(s1[0]);
                int v = Integer.valueOf(s1[1]);
                arr[u].put(v,1);
                arr[v].put(u,1);
            }

            long[] ans  = maxBeauty(arr,lv,rv,1,0,vis);

            //System.out.println(maxBeauty(graph,lvrv,1,0,0));
            //System.out.println(maxBeauty(graph,lvrv,1,1,0));

            pw.println(Math.max(ans[1],ans[0]));
            tc--;
        }
        pw.close();
    }
}
