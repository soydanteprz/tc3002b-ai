import java.io.*;
import java.util.*;

public class ddd {
    //----------------------------START--------------------------------------//
    public static void main(String[] args)
            throws IOException {

        ADMIN_MODE();

        int t = sc.ni();while(t-->0)
            solve();


    }

    static boolean[] neg;

    static void solve() throws IOException {
        int n = sc.ni();
        neg = new boolean[n];
        pr[] arr = new pr[n];
        for(int i = 0; i < n; i++) {
            long curr = sc.nl();
            if(curr < 0) neg[i] = true;
            arr[i] = new pr(Math.abs(curr), i);
        }
        Arrays.sort(arr);
        long[] res = new long[n];
        for(int i = 0; i < n-1; i+=2) {
            long lcm = lcm(arr[i].val, arr[i+1].val);
            res[arr[i].idx] = lcm/arr[i].val * (neg[arr[i].idx] ? -1 :1) * -1;
            res[arr[i+1].idx] = lcm/arr[i+1].val * (neg[arr[i+1].idx] ? -1 :1);
        }

        if(n%2==1) {
            long aa = arr[n-3].val * (neg[arr[n-3].idx] ? -1 : 1);
            long bb = arr[n-2].val * (neg[arr[n-2].idx] ? -1 : 1);
            long cc = arr[n-1].val * (neg[arr[n-1].idx] ? -1 : 1);

            if (aa == -1*bb) {
                if(aa == cc) {
                    pr temp = arr[n-1];
                    arr[n-1] = arr[n-2];
                    arr[n-2] = temp;
                } else {
                    pr temp = arr[n-3];
                    arr[n-3] = arr[n-1];
                    arr[n-1] = temp;
                }
            }
        }

        if(n%2==1) {
            long a = arr[n-2].val * (neg[arr[n-2].idx] ? -1 :1) +
                    arr[n-3].val * (neg[arr[n-3].idx] ? -1 :1);
            boolean f = (a < 0);
            a = Math.abs(a);
            long b = arr[n-1].val;
            long lcm = lcm(a, b);
            if(f) {
                res[arr[n-2].idx] = lcm/a * -1;
                res[arr[n-3].idx] = lcm/a * -1;
            } else {
                res[arr[n - 2].idx] = lcm / a;
                res[arr[n - 3].idx] = lcm / a;
            }
            res[arr[n-1].idx] = lcm/b * (neg[arr[n-1].idx] ? -1 :1) * - 1;
        }

        for(int i = 0; i < n; i++) {
            System.out.println(res[i]+" ");
        }
        System.out.println();
    }

    static class pr implements Comparable<pr>{
        long val;int idx;

        public pr(long val, int idx) {
            this.val = val;
            this.idx = idx;
        }

        
    }

    //--------------------------INPUT READER--------------------------------//

    //-----------------------------------------------------------------------//

    //---------------------------PRINTER-------------------------------------//
    //------------------------------------------------------------------------//

    //--------------------------VARIABLES------------------------------------//
    static FastReader sc = new FastReader();
    static OutputStream outputStream = System.out;
    //-----------------------------------------------------------------------//

    //--------------------------ADMIN_MODE-----------------------------------//
    private static void ADMIN_MODE() throws IOException {
        if (System.getProperty("ONLINE_JUDGE") == null) {
            sc = new FastReader(new FileInputStream("input.txt"));
        }
    }
    //-------------------------------UTILS------------------------------------//
    static long gcd(long a, long b) {
        return b == 0 ? (a < 0L ? -a: a) : gcd(b, a%b);
    }

    static long lcm(long a, long b) {
        return a/gcd(a, b)*b;
    }

    // region pow(a, b, mod) -> a^b
    
    // endregion
}
