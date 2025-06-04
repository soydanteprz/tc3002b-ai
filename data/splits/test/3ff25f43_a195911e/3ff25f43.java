import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

public class EDU121C {
    public static void main(String[] args) {
        JS scan = new JS();
        int t = scan.nextInt();
        while(t-->0){
            int n = scan.nextInt();
            long[] k = new long[n];
            long[] h = new long[n];
            for(int i =0;i<n;i++){
                k[i] = scan.nextInt();
            }
            for(int i = 0;i<n;i++){
                h[i] = scan.nextInt();
            }
            long ans = 0;
            long lastD = 0;
            long lastHP = 0;
            for(int i = 0;i<n;i++){
                long dist = k[i]-lastD;
                long lo = h[i];
                long hi = lastHP+dist;
                long hpComingOut = 0;
                while(lo<=hi){
                    long mid = (lo+hi)/2;
                    boolean check = false;
                    for(int j = i+1;j<n;j++){
                        if(k[j]-k[i]<h[j]-mid){
                            check = true;
                            break;
                        }
                    }
                    if(check){
                        //we're in trouble
                        lo = mid+1;
                    }else{
                        hi = mid-1;
                        hpComingOut = mid;
                    }
                }
                if(hpComingOut>dist){
                    ans+=summ(dist+lastHP)-summ(lastHP);
                    lastHP+=dist;
                }else{
                    long diff = dist-hpComingOut;
                    ans+=summ(hpComingOut);
                    lastHP = hpComingOut;
                }
                lastD = k[i];
            }
            System.out.println(ans);
        }
    }
    static long summ(long a){
        return (a*(a+1))/2;
    }
    
}
