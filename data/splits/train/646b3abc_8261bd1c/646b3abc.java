import java.util.*;
import java.io.*;

public class Main {
    static FastScanner sc = new FastScanner(System.in);
    static PrintWriter pw = new PrintWriter(System.out);
    static StringBuilder sb = new StringBuilder();
    static long mod = (long) 1e9 + 7;

    public static void main(String[] args) throws Exception {
        int n = sc.nextInt();
        for(int i = 0; i < n; i++) solve();
        pw.flush();
    }

    public static void solve() {
        int n = sc.nextInt();
        char[] a = sc.next().toCharArray();
        char[] b = sc.next().toCharArray();
        boolean isFirst = false;
        boolean allOk = true;
        int a1 = 0;
        int a0 = 0;
        int b1 = 0;
        int b0 = 0;
        int[][] cnt1 = new int[2][2];
        for(int i = 0; i < n; i++){
            if(a[i] != b[i]){
                allOk = false;
            }
            if(a[i] == '1'){
                a1++;
                isFirst = true;
            }else{
                a0++;
            }
            if(b[i] == '1'){
                b1++;
            }else{
                b0++;
            }
            cnt1[a[i]-'0'][b[i]-'0']++;
        }
        if(allOk){
            pw.println(0);
            return;
        }else if(!isFirst){
            pw.println(-1);
            return;
        }else if(a1 != b1 && (a0+1) != b1){
            pw.println(-1);
            return;
        }else{
            int same = cnt1[1][1] - cnt1[0][0] == 1 ? cnt1[1][1] + cnt1[0][0] : Integer.MAX_VALUE;
            int odd = cnt1[1][0] - cnt1[0][1] == 0 ? cnt1[1][0] + cnt1[0][1] : Integer.MAX_VALUE;
            int min = Math.min(same,odd);
            pw.println(min == Integer.MAX_VALUE ? -1 : min);
        }
    }
    

    
}


