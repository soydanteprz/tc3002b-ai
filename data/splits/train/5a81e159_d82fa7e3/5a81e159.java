import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;

public class C_CF {

    public static void main(String[] args) {
        FastScanner57 fs = new FastScanner57();
        PrintWriter pw = new PrintWriter(System.out);
        int t = fs.ni();
        //int t = 1;
        for (int tc = 0; tc < t; tc++) {
            int n = fs.ni();
            long[][] lr = new long[n][2];
            for (int i = 0; i < n; i++) {
                lr[i][0] = fs.nl();
                lr[i][1] = fs.nl();
            }
            List<List<Integer>> list = new ArrayList();
            for (int i = 0; i < n;i ++) {
                List<Integer> temp = new ArrayList();
                list.add(temp);
            }
            for (int i = 0; i < n-1; i++) {
                int a = fs.ni()-1, b = fs.ni()-1;
                list.get(a).add(b);
                list.get(b).add(a);
            }
            Long[][] dp = new Long[n][2];
            pw.println(recur(0,0,-1,new long[] {0,0},dp,lr,list));
        }
        pw.close();

    }
    // 0 -> left was chosen
    // 1 -> right was chosen
    public static long recur(int ind, int p,int prev, long[] v, Long[][] dp, long[][] lr,List<List<Integer>> list) {

        long last = v[0];
        long ls = 0L;
        long rs = 0L;
        if (p==1) {
            last = v[1];
        }
        if (ind!=0) ls += (long)Math.abs(last-lr[ind][0]);
        if (ind!=0) rs += (long)Math.abs(last-lr[ind][1]);
        if (dp[ind][p]!=null) return dp[ind][p];
        long[] cur = lr[ind];
        
        List<Integer> temp = list.get(ind);
        for (int val : temp) {
            if (prev==val) continue;
            ls += recur(val,0,ind,cur,dp,lr,list);
            rs += recur(val,1,ind,cur,dp,lr,list);
        }
        return dp[ind][p] = Math.max(ls,rs);
    }

    

    

}




