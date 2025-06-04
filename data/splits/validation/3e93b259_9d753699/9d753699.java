import java.util.*;
import java.io.*;
import java.math.*;

public class C1499 {
    public static void main(String[] args) throws IOException {
        PrintWriter out = new PrintWriter(System.out);
        Reader s = new Reader();
        int t = s.i();
        while (t-- > 0) {
            int n = s.i();
            long [] arr = s.arrLong(n);
            long odd = arr[1] , even = arr[0];
            long [] a = new long[n];
            a[0] = arr[0]*n;
            a[1] = arr[1]*n;
            int in1 = 0 , in2 = 1;
            long ans = a[0]+a[1];
            for (int i=2;i<n;i++) {
                if (i%2 == 0) {
                    long val = even + (n-i/2)*arr[i];
                    long val2 = a[i-2] - arr[in1] + arr[i];
                    if (val < val2) in1 = i;
                    a[i] = Math.min(val2,val);
                    even += arr[i];
                } else {
                    long val = odd + (n-i/2)*arr[i];
                    long val2 = a[i-2] - arr[in2] + arr[i];
                    if (val < val2) in2 = i;
                    a[i] = Math.min(val2,val);
                    odd += arr[i];
                }
                ans = Math.min(ans,a[i]+a[i-1]);
            }
            out.println(ans);
        }
        out.flush();
    }

    

    

    

    

    

    

    

    

    

    
//    static class pairLong implements Comparator<pairLong> {
//        long first, second;
//
//        pairLong() {
//        }
//
//        pairLong(long first, long second) {
//            this.first = first;
//            this.second = second;
//        }
//
//        @Override
//        public int compare(pairLong p1, pairLong p2) {
//            if (p1.first == p2.first) {
//                if(p1.second > p2.second) return 1;
//                else return -1;
//            }
//            if(p1.first > p2.first) return 1;
//            else return -1;
//        }
//    }
//    static class pair implements Comparator<pair> {
//        int first, second;
//
//        pair() {
//        }
//
//        pair(int first, int second) {
//            this.first = first;
//            this.second = second;
//        }
//
//        @Override
//        public int compare(pair p1, pair p2) {
//            if (p1.first == p2.first) return p1.second - p2.second;
//            return p1.first - p2.first;
//        }
//    }
}
