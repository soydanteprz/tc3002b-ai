import java.io.*;
import java.util.*;

public class q3 {
    public static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
//    public static long mod = 1000000007;

    public static void solve() throws Exception {
        String[] parts = br.readLine().split(" ");
        int n = Integer.parseInt(parts[0]);


        int[] arr = new int[n + 1];
        boolean[] vis = new boolean[n + 1];

        for(int i = 1;i <= n;i++){
            if(vis[i]) continue;

            ArrayList<Integer> list = new ArrayList<>();
            while(true){
                System.out.println("? " + i);
                int val = Integer.parseInt(br.readLine());
                vis[val] = true;
                list.add(val);

                if(list.size() > 1 && val == list.get(0)) break;
            }

            for(int j = 0;j < list.size() - 1;j++) arr[list.get(j)] = list.get(j + 1);
//            arr[list.get(list.size() - 1)] = list.get(0);
        }

        System.out.print("! ");
        StringBuilder ans = new StringBuilder();
        for(int i = 1;i <= n;i++) ans.append(arr[i]).append(" ");
        System.out.println(ans);


    }

    public static void main(String[] args) throws Exception {
        int tests = Integer.parseInt(br.readLine());
        for (int test = 1; test <= tests; test++) {
            solve();
        }
    }

//    public static ArrayList<Integer> primes;
//    public static void seive(int n){
//        primes = new ArrayList<>();
//        boolean[] arr = new boolean[n + 1];
//        Arrays.fill(arr,true);
//
//        for(int i = 2;i * i <= n;i++){
//            if(arr[i]) {
//                for (int j = i * i; j <= n; j += i) {
//                    arr[j] = false;
//                }
//            }
//        }
//        for(int i = 2;i <= n;i++) if(arr[i]) primes.add(i);
//    }


//    public static void sort(int[] arr){
//        ArrayList<Integer> temp = new ArrayList<>();
//        for(int val : arr) temp.add(val);
//
//        Collections.sort(temp);
//
//        for(int i = 0;i < arr.length;i++) arr[i] = temp.get(i);
//    }

//    public static void sort(long[] arr){
//        ArrayList<Long> temp = new ArrayList<>();
//        for(long val : arr) temp.add(val);
//
//        Collections.sort(temp);
//
//        for(int i = 0;i < arr.length;i++) arr[i] = temp.get(i);
//    }
//
//    public static long power(long a,long b,long mod){
//        if(b == 0) return 1;
//
//        long p = power(a,b / 2,mod);
//        p = (p * p) % mod;
//
//        if(b % 2 == 1) return (p * a) % mod;
//        return p;
//    }
//    public static long modDivide(long a,long b,long mod){
//        return ((a % mod) * (power(b,mod - 2,mod) % mod)) % mod;
//    }
//
//    public static int GCD(int a,int b){
//        return b == 0 ? a : GCD(b,a % b);
//    }
//    public static long GCD(long a,long b){
//        return b == 0 ? a : GCD(b,a % b);
//    }
//
//    public static int LCM(int a,int b){
//        return a * b / GCD(a,b);
//    }
//    public static long LCM(long a,long b){
//        return a * b / GCD(a,b);
//    }
}
