import java.io.*;
import java.util.*;

public class T4  {

    public static void main(String[] args) {
        FastScanner s = new FastScanner();
        T4 main = new T4();
        int n = 1;
        while (n-- > 0) {
            main.MainPrint(s);
        }
    }

    long mod =  998244353L;

    private void MainPrint(FastScanner s) {
        int n = s.nextInt();
        k = s.nextInt();
        TreeMap<Integer, int[]> map = new TreeMap<>();
        int l, r;
        int[] get;
        for (int i = 0; i < n; i++) {
            l = s.nextInt();
            r = s.nextInt();
            if (!map.containsKey(l)) {
                get = new int[2];
                get[0]++;
                map.put(l, get);
            } else {
                get = map.get(l);
                get[0]++;
            }
            if (!map.containsKey(r + 1)) {
                get = new int[2];
                get[1]++;
                map.put(r + 1, get);
            } else {
                get = map.get(r + 1);
                get[1]++;
            }
        }
        int count = 0;
        long ans = 0;
        precomp();
        for (int i : map.keySet()) {
            get = map.get(i);
            count += get[0] - get[1];
            ans += rem[count] - rem[count-get[0]];
            ans %= mod;
        }
        if (ans<0) ans+=mod;
        System.out.println(ans%mod);
    }


    long mul(long a, long b) {
        return a*b%mod;
    }

    long exp(long base, long e) {
        if (e==0) return 1;
        long half=exp(base, e/2);
        if (e%2==0) return mul(half, half);
        return mul(half, mul(half, base));
    }

    long modInv(long x) {
        return exp(x, mod-2);
    }

    long[] facts;
    long[] factInvs;
    long[] rem;
    int k;

    void precomp() {
        facts=new long[300_001];
        factInvs=new long[300_001];
        rem = new long[300_001];
        factInvs[0]=facts[0]=1;

        for (int i=1; i<facts.length; i++)
            facts[i]=mul(facts[i-1], i);

        factInvs[facts.length-1]=modInv(facts[facts.length-1]);

        for (int i=facts.length-2; i>=0; i--)
            factInvs[i]=mul(factInvs[i+1], i+1);
        for (int i = k;i<rem.length;i++){
            rem[i] =mul(facts[i], mul(factInvs[k], factInvs[i-k]));
        }
    }

    

}
