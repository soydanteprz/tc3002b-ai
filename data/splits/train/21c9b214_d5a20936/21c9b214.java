import java.util.*;
import java.io.*;
 
public class AiseHi {
	static Scanner sc = new Scanner(System.in);
	public static void main (String[] args) {
		Scanner sc = new Scanner(System.in);
		int t = sc.nextInt();
		z : while(t-->0) {
			int n = sc.nextInt();
			char sadsasa[] = sc.next().toCharArray();
			char sadsdsdsasa[] = sc.next().toCharArray();
			
			if(sadsdscds(sadsasa, sadsdsdsasa)) {
				System.out.print("0\n");
				continue;
			}
			
			int sadsdscdhggs = 0, sadsdscdiyuihggs = 0, sadsdscdiyhtruihggs = 0, poinhnhgs = 0;
			
			for(int i=0;i<n;i++) {
				if(sadsasa[i]!=sadsdsdsasa[i]) {
					if(sadsasa[i] == '0') poinhnhgs++;
					else sadsdscdiyhtruihggs++;
				}
				else {
					if(sadsasa[i] == '0') sadsdscdiyuihggs++;
					else sadsdscdhggs++;
				}
			}
			
			int rtenymu = 0;
			
		    rtenymu = uiuyfdsfrinhgw(sadsdscdhggs, sadsdscdiyuihggs, sadsdscdiyhtruihggs, poinhnhgs);

			rtenymu = Math.min(rtenymu, uiuyfdsfrinhgwdss(sadsdscdhggs, sadsdscdiyuihggs, sadsdscdiyhtruihggs, poinhnhgs));
			
			
			if(rtenymu == Integer.MAX_VALUE) {
				rtenymu = -1;
			}
			
			System.out.print(rtenymu+"\n");
		}
	}
	private static int uiuyfdsfrinhgwdss(int sadsdscdhggs, int sadsdscdiyuihggs, int sadsdscdiyhtruihggs, int poinhnhgs) {
		boolean rtenymuefwefw = false;
		int uiuyinhgw = 0;
		while(true) {
			if(sadsdscdiyhtruihggs == 0 && poinhnhgs == 0) return uiuyinhgw;
			if((rtenymuefwefw && sadsdscdhggs == 0) || (!rtenymuefwefw && sadsdscdiyhtruihggs == 0)) return Integer.MAX_VALUE;
			
			if(rtenymuefwefw) {
				int td1 = sadsdscdiyuihggs;
				int td0 = sadsdscdhggs - 1;
				int te1 = poinhnhgs + 1;
				int te0 = sadsdscdiyhtruihggs;
				
				sadsdscdiyhtruihggs = td1;
				poinhnhgs = td0;
				sadsdscdhggs = te1;
				sadsdscdiyuihggs = te0;
			}
			else {
				int td1 = 1 + sadsdscdiyuihggs;
				int td0 = sadsdscdhggs;
				int te1 = poinhnhgs;
				int te0 = sadsdscdiyhtruihggs-1;
				
				sadsdscdiyhtruihggs = td1;
				poinhnhgs = td0;
				sadsdscdhggs = te1;
				sadsdscdiyuihggs = te0;
			}
			
			rtenymuefwefw = !rtenymuefwefw;
			uiuyinhgw++;
		}
	}
	private static int uiuyfdsfrinhgw(int sadsdscdhggs, int sadsdscdiyuihggs, int sadsdscdiyhtruihggs, int poinhnhgs) {
		boolean rtenymuefwefw = true;
		int uiuyinhgw = 0;
		while(true) {
			if(sadsdscdiyhtruihggs == 0 && poinhnhgs == 0) return uiuyinhgw;
			if((rtenymuefwefw && sadsdscdhggs == 0) || (!rtenymuefwefw && sadsdscdiyhtruihggs == 0)) return Integer.MAX_VALUE;
			
			if(rtenymuefwefw) {
				int td1 = sadsdscdiyuihggs;
				int td0 = sadsdscdhggs - 1;
				int te1 = poinhnhgs + 1;
				int te0 = sadsdscdiyhtruihggs;
				
				sadsdscdiyhtruihggs = td1;
				poinhnhgs = td0;
				sadsdscdhggs = te1;
				sadsdscdiyuihggs = te0;
			}
			else {
				int td1 = 1 + sadsdscdiyuihggs;
				int td0 = sadsdscdhggs;
				int te1 = poinhnhgs;
				int te0 = sadsdscdiyhtruihggs-1;
				
				sadsdscdiyhtruihggs = td1;
				poinhnhgs = td0;
				sadsdscdhggs = te1;
				sadsdscdiyuihggs = te0;
			}
			
			rtenymuefwefw = !rtenymuefwefw;
			uiuyinhgw++;
		}
	}
	private static boolean sadsdscds(char[] sadsasa, char[] sadsdsdsasa) {
		for(int i=0;i<sadsasa.length;i++) {
			if(sadsasa[i]!=sadsdsdsasa[i]) return false;
		}
		return true;
	}
	
	
	
	
	
	
	 
	
	
	
	
  
    
//	private static boolean check(long n,long val) {
//		if(n == 0) return true;
//		
//		while(val<=n) {
////			if(check(n-val,))
//		}
//	}
	
	
	
	
//	static int ceil(int sadsasa,int sadsdsdsasa) {
//		return sadsasa/sadsdsdsasa + (sadsasa%sadsdsdsasa==0?0:1);
//	}
//	static int fac[] = new int[2000009];
	
	
//	static long gcd(long sadsasa,long sadsdsdsasa) { if(sadsdsdsasa==0) return sadsasa; return gcd(sadsdsdsasa,sadsasa%sadsdsdsasa); }
}
 



