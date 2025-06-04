import java.io.*;
import java.util.*;

public class B {
    //--------------------------INPUT READER---------------------------------//
    //-----------------------------------------------------------------------//

    //---------------------------PRINTER-------------------------------------//
    //-----------------------------------------------------------------------//

    //--------------------------VARIABLES------------------------------------//
    static FastReader sc = new FastReader();
    //-----------------------------------------------------------------------//

    //--------------------------ADMIN_MODE-----------------------------------//
    private static void ADMIN_MODE() throws IOException {
        if (System.getProperty("ONLINE_JUDGE") == null) {
            sc = new FastReader(new FileInputStream("input.txt"));
        }
    }
    //-----------------------------------------------------------------------//

    //----------------------------START--------------------------------------//
    public static void main(String[] args)
            throws IOException {

        ADMIN_MODE();

        int t = sc.ni();while(t-->0)
            solve();


    }

    static void solve() throws IOException {
        int n = sc.ni();
        long k = sc.nl();
        Long[] arr = new Long[n];
        for(int i = 0; i < n; i++) arr[i] = sc.nl();
        Arrays.sort(arr);
        long sum = 0;
        for(int i = 0; i < n; i++) {
            sum += arr[i];
        }
        if(sum <= k) {
            System.out.println(0);
            return;
        }


        long cont = 0;
        long min = sum-k;
        for(int i = n-1; i >= 0; i--) {
            cont += arr[i];
            long psum = sum-cont;


            if(psum <= k) {
                long extra = k-psum;
                if(arr[0]*(n-i)<=extra) {
                    min = Math.min(min, n-i);
                    continue;
                }

                if(i == 0) {
                    long q = k/n;
                    long ans = arr[0]-q+n-1;
                    min = Math.min(min, ans);
                    continue;
                }
                extra += arr[0];
                long q = extra/(n-i+1);
                long ans = arr[0]-q+n-i;
                min = Math.min(min, ans);

            } else {
                long toRem = psum-k-arr[0];
                long q = (long)Math.ceil(toRem/(n-i+1.0));
                long ans = q+arr[0]+n-i;
                min = Math.min(min, ans);
            }
        }

        System.out.println(min);
    }

}
