import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.*;


public class Cr766A {

    public static void main(String[] args) {

        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader sc = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Solver solver = new Solver();
        int t = sc.nextInt();
//        int t = 1;
        while (t-- != 0) {
            solver.solve(sc, out);
        }
        out.close();

    }

    static class Solver {
        ArrayList<int[]>[] gp;
        int[] ans;
        public void solve(InputReader sc, PrintWriter out) {
            int n = sc.nextInt();
            gp = new ArrayList[n];
            ans = new int[n-1];
            for (int i = 0; i < n; i++) {
                gp[i] = new ArrayList<>();
            }
            for (int i = 0; i < n - 1; i++) {
                int u = sc.nextInt()-1;
                int v = sc.nextInt()-1;
                gp[u].add(new int[] {v,i});
                gp[v].add(new int[] {u,i});
            }
            for(int i = 0; i < n; i++) {
                if(gp[i].size()>2) {
                    out.println(-1);
                    return;
                }
            }
            int oneDegree = -1;
            for(int i = 0; i < n; i++) {
                if(gp[i].size()==1) {
                    oneDegree = i;
                }
            }
            if(oneDegree==-1) {
                throw new RuntimeException();
            }
            dfs(oneDegree,-1,3);
            for(int i = 0; i < n-1; i++) {
                if(i>0) out.print(" ");
                out.print(ans[i]);
            }
            out.println();
        }

        private void dfs(int src, int p, int prevPrime) {
            int ocr = 0;
            for(int[] vs : gp[src]) {
                int v = vs[0];
                int idx = vs[1];
                if(v==p) continue;
                ocr++;
                ans[idx] = 5-prevPrime;
                dfs(v,src,ans[idx]);
            }
            if(ocr>1) {
                throw new RuntimeException();
            }
        }

        

    }

    

    

    

    

    


}
