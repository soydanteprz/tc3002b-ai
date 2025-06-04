import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.*;



public class Cgr18A {


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
        final int INF = (int) 1e9;
        public void solve(InputReader sc, PrintWriter out) {
            int n = sc.nextInt();
            char[] a = sc.next().toCharArray();
            char[] b = sc.next().toCharArray();
            boolean isAllZeroA = isAllZero(a);
            boolean isAllZeroB = isAllZero(b);
            if(isAllZeroA && isAllZeroB) {
                out.println(0);
                return;
            }
            if(isAllZeroA || isAllZeroB) {
                out.println(-1);
                return;
            }
            int eq = 0;
            for(int i = 0; i< n;i++) if(a[i]==b[i]) eq++;
            int eqHasOne = 0;
            int notEqHasOne = 0;
            int noteq = n-eq;
            for (int i = 0; i < n; i++) {
                if(a[i]==b[i] && a[i]=='1') eqHasOne++;
                if(a[i]!=b[i] && a[i]=='1') notEqHasOne++;
            }
            int l = solveIt(new int[] {eq,noteq},new int[] {eqHasOne,notEqHasOne},0);
            int r = solveIt(new int[] {eq,noteq},new int[] {eqHasOne,notEqHasOne},1);
            if(l==INF && r==INF) {
                out.println(-1);
            } else {
                out.println(Math.min(l,r));
            }
        }

        private boolean isAllZero(char[] a) {
            for(char c : a) if(c=='1') return false;
            return true;
        }

        private int solveIt(int[] arr,int[] hasOne, int minusPointer) {
            if(arr[0]==0 && arr[1]==0) throw new RuntimeException();
            int cnt = 0;
            int max = arr[0] + arr[1];
            while(arr[1]!=0) {
                int minus = minusPointer;
                int plus = 1-minus;
                if(hasOne[minusPointer]<1) return INF;
                int eqHasOne = hasOne[0];
                int notEqHasOne = hasOne[1];
                int eqHasZero = arr[0] - eqHasOne;
                int notEqHasZero = arr[1] - notEqHasOne;
                if(minusPointer==0) {
                    int newNotEqHasOne = eqHasZero;
                    int newEqHasOne = notEqHasZero + 1;
                    hasOne[0] = newEqHasOne;
                    hasOne[1] = newNotEqHasOne;
                } else {
                    int newEqHasOne = notEqHasZero;
                    int newNotEqHasOne = eqHasZero + 1;
                    hasOne[0] = newEqHasOne;
                    hasOne[1] = newNotEqHasOne;
                }
                int minusValue = arr[minus]-1;
                int plusValue = arr[plus]+1;
                arr[minus] = plusValue;
                arr[plus] = minusValue;
                minusPointer = 1-minusPointer;
                cnt++;
                if(cnt>max+10) break;
            }
            if(arr[1]!=0) return INF;
            return cnt;
        }
    }

    

    

    

    

    


}
