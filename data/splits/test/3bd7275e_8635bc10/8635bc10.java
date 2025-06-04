import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.StringTokenizer;




public class C{

//    static int N = 200010;
    static int n;
    static long k;
    static long[] a;
    static long[] s; // 求出a排序之后的前缀和

    public static void main(String[] args){

//        Scanner sc = new Scanner(System.in);
        FastScanner sc = new FastScanner();
        int T = sc.nextInt();
        while(T-- > 0){
            n = sc.nextInt();
            k = sc.nextLong();
//            a = sc.readArray(n);
            a = new long[n + 1];
            for(int i = 1; i <= n; i++) a[i] = sc.nextInt();
            System.out.println(slover());
        }
    }

    static long slover(){

        Arrays.sort(a);
        // for(int i = 1; i <= n; i++) System.out.print(a[i] + " ");
        // System.out.println();
        s = new long[n + 1];
        for(int i = 1; i <= n; i++) s[i] = s[i - 1] + a[i];
        // System.out.println(s[n]);
        long res = (long)1e10;
        // 能枚举进行赋值操作的元素个数
        for(int i = 1; i <= n; i++){
//            long selected = (n - i) * a[1]; // 最小的替换(n - i)
            long t;
            if(s[i] + (n-i)*a[1] - k <= 0) t = 0;
            else t = (long) Math.ceil((s[i] + (n-i) * a[1] - k) * 1.0 / (n-i+1));
//            if(selected + s[i] <= k){
//                t = 0;
//            }else{
//                // System.out.println("get in");
//                t = (long)Math.ceil((s[i] + selected - k) * 1.0 / (1 + n - i));
//            }

            res = Math.min(res, n - i + t);
            // System.out.println((n - i) + " -> " + (s[i] + selected - k) + " " + (1 + n - i) + " ->" + res);
        }
        return res;
    }

    

}
