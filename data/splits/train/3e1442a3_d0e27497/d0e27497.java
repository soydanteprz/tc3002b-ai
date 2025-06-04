import java.util.*;
import java.io.*;

public class Main {
    static FastScanner sc = new FastScanner(System.in);
    static PrintWriter pw = new PrintWriter(System.out);
    static StringBuilder sb = new StringBuilder();
    static long mod = (long) 1e9 + 7;

    public static void main(String[] args) throws Exception {
        int t = sc.nextInt();
        for(int i = 0; i < t; i++) solve();
        pw.flush();
    }
    
    static ArrayList<ArrayList<int[]>> map;
    static int[] ans;
    public static void solve() {
        sb.setLength(0);
        int n = sc.nextInt();
        ans = new int[n-1];
        map = new ArrayList<>();
        for(int i = 0; i < n; i++){
            map.add(new ArrayList<>());
        }
        int[] cnt = new int[n];
        for(int i = 0; i < n-1; i++){
            int[] e = sc.nextIntArray(2);
            e[0]--;
            e[1]--;
            cnt[e[0]]++;
            cnt[e[1]]++;
            //edge.add(e);
            map.get(e[0]).add(new int[]{e[1],i});
            map.get(e[1]).add(new int[]{e[0],i});
        }
        for(int i = 0; i < n; i++){
            if(cnt[i] > 2){
                pw.println(-1);
                return;
            }
        }
        
        for(int i = 0; i < n; i++){
            if(cnt[i] == 1){
                ArrayDeque<int[]> dq = new ArrayDeque<>();
                boolean[] used = new boolean[n];
                used[i] = true;
                dq.add(new int[]{i,0});
                while(dq.size() > 0){
                    int[] now = dq.poll();
                    int u = now[0];
                    int c = now[1];
                    //System.err.println(u + " " + c + " "+ map.get(u).size());
                    for(int[] next : map.get(u)){
                        int v = next[0];
                        int ei = next[1];
                        //System.err.println(u + " " + v + " " + c + " " + ei);
                        if(!used[v]){
                            used[v] = true;
                            //System.err.println(u + " " + v + " " + c);
                            ans[ei] = c % 2 == 0 ? 3 : 2;
                            dq.add(new int[]{v,c+1});
                        }
                    }
                }
                for(int v : ans){
                    if(v == 0){
                        pw.println(-1);
                        return;
                    }
                    sb.append(v).append(" ");
                }
                pw.println(sb.toString().trim());
                return;
            }
        }
        pw.println(-1);
    }

    
}


